"""Every backtest figure: one folder per variant plus cross-variant comparisons -> figures/.

Reads results/<side>_lb<lookback>/ written by backtest.py and robustness.py (run_all.sh runs all).
"ML average" = an equal blend of the seven ML portfolios (daily returns averaged), so no model
is picked with hindsight.

    python 08_markowitz/plot_backtest.py
"""
import sys
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.plotting import INK, INK_2, SERIES, SEQUENTIAL, DIVERGING, plt, save  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
MODELS = ["Ridge", "ElasticNet", "Random Forest", "XGBoost", "RNN", "LSTM", "ARIMA"]
COLOR = {**dict(zip(MODELS, SERIES)), "Historical": INK_2, "ML average": SERIES[7], "Benchmark": INK}
VARIANTS = [("long", 252), ("long", 20), ("longshort", 252), ("longshort", 20)]
WEIGHTING = {"MV": "Markowitz max-Sharpe", "EW": "equal weight"}


def label(side, lb):
    return f"{'Long-only' if side == 'long' else 'Long-short'}, {lb}-day lookback"


def stats(r):
    eq = (1 + r).cumprod()
    return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252), "max DD": (eq / eq.cummax() - 1).min()}


def style(name):
    if name == "Historical":
        return {"color": COLOR[name], "lw": 2.4, "ls": "--"}
    if name == "ML average":
        return {"color": COLOR[name], "lw": 2.6}
    if name == "Benchmark":
        return {"color": COLOR[name], "lw": 1.1, "ls": ":"}
    return {"color": COLOR[name], "lw": 1.1, "alpha": 0.75}


def date_ticks(*axes):
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def legend_below(ax, ncol=5):
    ax.legend(ncol=ncol, loc="upper left", bbox_to_anchor=(0, -0.1))


def hbars(ax, v, fmt, title, ref=None):
    y = np.arange(len(v))
    ax.barh(y, v.values, color=[COLOR.get(n, INK_2) for n in v.index], height=0.6)
    ax.set_yticks(y, v.index)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, lw=0.8)
    if ref is not None:
        ax.axvline(ref, color=INK_2, lw=1, ls="--")
    for i, x in enumerate(v.values):
        ax.text(x, i, " " + fmt.format(x) + " ", va="center", ha="left" if x >= 0 else "right", fontsize=8, color=INK_2)
    ax.set_title(title, loc="left")
    ax.margins(x=0.18)
    ax.grid(axis="y", visible=False)


def load(side, lb):
    d = RES / f"{side}_lb{lb}"
    rets = pd.read_parquet(d / "daily_returns.parquet")
    for wt in WEIGHTING:
        rets[f"ML average {wt}"] = rets[[f"{m} {wt}" for m in MODELS]].mean(axis=1)
    return {"dir": d, "rets": rets, "periods": pd.read_csv(d / "periods.csv", parse_dates=["Date"]),
            "legs": pd.read_parquet(d / "leg_returns.parquet"),
            "runs": pd.read_csv(d / "offset_runs.csv") if (d / "offset_runs.csv").exists() else None}


def variant_figures(side, lb, v):
    out = FIG / f"{side}_lb{lb}"
    rets, per, title = v["rets"], v["periods"], label(side, lb)
    sources = ["Historical", *MODELS, "ML average"]

    # 01 equity curves, Markowitz and equal weight
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, wt in zip(axes, WEIGHTING):
        for s in [*MODELS, "Historical", "ML average"]:
            eq = (1 + rets[f"{s} {wt}"]).cumprod()
            ax.plot(eq.index, eq, label=s, **style(s))
        ax.plot(rets.index, (1 + rets["Benchmark"]).cumprod(), label="Benchmark (market)", **style("Benchmark"))
        ax.axhline(1, color=INK_2, lw=0.6)
        ax.set_title(f"Growth of 1, {WEIGHTING[wt]} ({title})", loc="left")
    legend_below(axes[1])
    date_ticks(*axes)
    save(fig, out / "01_equity_curves.png")

    # 02 drawdowns (underwater)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for s in [*MODELS, "Historical", "ML average", "Benchmark"]:
        r = rets["Benchmark"] if s == "Benchmark" else rets[f"{s} MV"]
        eq = (1 + r).cumprod()
        ax.plot(eq.index, (eq / eq.cummax() - 1) * 100, label=s, **style(s))
    ax.set_ylabel("drawdown from peak (%)")
    ax.set_title(f"Drawdowns, Markowitz ({title})", loc="left")
    legend_below(ax)
    date_ticks(ax)
    save(fig, out / "02_drawdowns.png")

    # 03 ML (average and range of the 7 models) vs historical
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, wt in zip(axes, WEIGHTING):
        eqs = pd.DataFrame({m: (1 + rets[f"{m} {wt}"]).cumprod() for m in MODELS})
        ax.fill_between(eqs.index, eqs.min(axis=1), eqs.max(axis=1), color=COLOR["ML average"], alpha=0.15,
                        lw=0, label="range of the 7 ML models")
        for s in ["ML average", "Historical"]:
            ax.plot(eqs.index, (1 + rets[f"{s} {wt}"]).cumprod(), label=s, **style(s))
        ax.plot(eqs.index, (1 + rets["Benchmark"]).cumprod(), label="Benchmark", **style("Benchmark"))
        ax.set_title(f"ML vs historical μ, {WEIGHTING[wt]}", loc="left")
    legend_below(axes[0], 4)
    date_ticks(*axes)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    save(fig, out / "03_ml_vs_historical.png")

    # 04 rolling 6-month Sharpe
    fig, ax = plt.subplots(figsize=(12, 4))
    for s in ["Historical", "ML average", "Benchmark"]:
        r = rets["Benchmark"] if s == "Benchmark" else rets[f"{s} MV"]
        rs = r.rolling(126).mean() / r.rolling(126).std() * np.sqrt(252)
        ax.plot(rs.index, rs, label=s, **style(s))
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("Sharpe (126-day window)")
    ax.set_title(f"Rolling 6-month Sharpe, Markowitz ({title})", loc="left")
    legend_below(ax, 3)
    date_ticks(ax)
    save(fig, out / "04_rolling_sharpe.png")

    # 05 distribution of 20-day period returns
    mv = per[per.weighting == "MV"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    data = [mv[mv.source == s].period_return * 100 for s in ["Historical", *MODELS]]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK, "lw": 1.5},
                    flierprops={"marker": "o", "markersize": 4, "markerfacecolor": INK_2, "markeredgecolor": "none"})
    for patch, s in zip(bp["boxes"], ["Historical", *MODELS]):
        patch.set(facecolor=COLOR[s], alpha=0.55, edgecolor=COLOR[s])
    for i, s in enumerate(["Historical", *MODELS], 1):
        ax.scatter(i, mv[mv.source == s].period_return.mean() * 100, marker="D", s=40, color=INK, zorder=3,
                   label="mean" if i == 1 else None)
    ax.set_xticks(range(1, len(MODELS) + 2), ["Historical", *MODELS])
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("20-day return (%)")
    ax.set_title(f"20-day holding-period returns, Markowitz ({title})", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, out / "05_period_return_distribution.png")

    # 06 expected vs realised 20-day return: how overconfident is each mu?
    ml = mv[mv.source.isin(MODELS)].groupby("Date")[["expected_return", "period_return"]].mean()
    hist = mv[mv.source == "Historical"].set_index("Date")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(hist.index, hist.expected_return * 100, label="Historical: expected", **style("Historical"))
    ax.plot(ml.index, ml.expected_return * 100, label="ML average: expected", **style("ML average"))
    ax.plot(hist.index, hist.period_return * 100, color=COLOR["Historical"], lw=1, marker="o", ms=4, alpha=0.6,
            label="Historical: realised")
    ax.plot(ml.index, ml.period_return * 100, color=COLOR["ML average"], lw=1, marker="o", ms=4, alpha=0.6,
            label="ML average: realised")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("portfolio 20-day return (%)")
    ax.set_title("Expected (Markowitz input) vs realised return, per rebalance", loc="left")
    legend_below(ax, 2)
    date_ticks(ax)
    ax = axes[1]
    x = np.arange(2)
    exp_ = [hist.expected_return.mean() * 100, ml.expected_return.mean() * 100]
    real = [hist.period_return.mean() * 100, ml.period_return.mean() * 100]
    ax.bar(x - 0.2, exp_, 0.38, color=[COLOR["Historical"], COLOR["ML average"]], alpha=0.45, label="expected")
    ax.bar(x + 0.2, real, 0.38, color=[COLOR["Historical"], COLOR["ML average"]], label="realised")
    for xi, e, r in zip(x, exp_, real):
        ax.text(xi - 0.2, e, f"{e:.1f}%", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.text(xi + 0.2, r, f"{r:.1f}%", ha="center", va="bottom" if r >= 0 else "top", fontsize=8, color=INK_2)
    ax.set_xticks(x, ["Historical", "ML average"])
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_title("Average over all rebalances", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    save(fig, out / "06_expected_vs_realised.png")

    # 07 calendar-year returns heatmap
    names = ["Historical", *MODELS, "ML average", "Benchmark"]
    yr = pd.DataFrame({s: (1 + (rets["Benchmark"] if s == "Benchmark" else rets[f"{s} MV"]))
                       .groupby(rets.index.year).prod() - 1 for s in names}).T * 100
    fig, ax = plt.subplots(figsize=(7, 5.5))
    lim = np.nanmax(np.abs(yr.values))
    ax.imshow(yr.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(yr.shape[1]), [f"{c}{' (part)' if c in (2024, 2026) else ''}" for c in yr.columns])
    ax.set_yticks(range(len(names)), names)
    for i in range(yr.shape[0]):
        for j in range(yr.shape[1]):
            ax.text(j, i, f"{yr.iat[i, j]:+.1f}%", ha="center", va="center", fontsize=9, color=INK)
    ax.grid(False)
    ax.set_title(f"Return by calendar year, Markowitz\n({title})", loc="left")
    save(fig, out / "07_annual_returns.png")

    # 08 risk vs return
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in ["Historical", *MODELS, "ML average"]:
        for wt, mk in (("MV", "s"), ("EW", "o")):
            st = stats(rets[f"{s} {wt}"])
            ax.scatter(st["vol"] * 100, st["CAGR"] * 100, marker=mk, s=70, color=COLOR[s], edgecolor="white", lw=1.5,
                       zorder=3)
            if wt == "MV":
                ax.annotate(s, (st["vol"] * 100, st["CAGR"] * 100), xytext=(6, 3), textcoords="offset points",
                            fontsize=8, color=INK_2)
    st = stats(rets["Benchmark"])
    ax.scatter(st["vol"] * 100, st["CAGR"] * 100, marker="*", s=180, color=INK, zorder=3)
    ax.annotate("Benchmark", (st["vol"] * 100, st["CAGR"] * 100), xytext=(6, 3), textcoords="offset points",
                fontsize=8, color=INK_2)
    ax.scatter([], [], marker="s", color=INK_2, label="Markowitz")
    ax.scatter([], [], marker="o", color=INK_2, label="equal weight")
    ax.set_xlabel("annualised volatility (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title(f"Risk vs return ({title})", loc="left")
    ax.legend(loc="lower right")
    save(fig, out / "08_risk_return.png")

    # 09 robustness: Sharpe on each of the 20 start days
    if v["runs"] is not None:
        runs = v["runs"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for ax, wt in zip(axes, WEIGHTING):
            data = [runs[(runs.source == s) & (runs.weighting == wt)].Sharpe for s in ["Historical", *MODELS]]
            bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK, "lw": 1.5})
            for patch, s in zip(bp["boxes"], ["Historical", *MODELS]):
                patch.set(facecolor=COLOR[s], alpha=0.55, edgecolor=COLOR[s])
            b = runs[runs.source == "Benchmark"].Sharpe.mean()
            ax.axhline(b, color=INK, lw=1, ls=":", label=f"Benchmark ({b:.2f})")
            ax.set_xticks(range(1, len(MODELS) + 2), ["Hist.", *[m.replace("Random Forest", "RF") for m in MODELS]])
            ax.set_title(f"Sharpe over the 20 start days, {WEIGHTING[wt]}", loc="left")
            ax.grid(axis="x", visible=False)
            ax.legend(loc="lower right")
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
        save(fig, out / "09_robustness_start_days.png")

    # 10 turnover and diversification of the Markowitz portfolios
    g = mv[mv.Date > mv.Date.min()].groupby("source")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    hbars(axes[0], (g.turnover.mean() * 100).reindex(["Historical", *MODELS]), "{:.0f}%",
          "Average turnover per rebalance (%)")
    hbars(axes[1], mv.groupby("source").eff_n.mean().reindex(["Historical", *MODELS]), "{:.1f}",
          "Effective number of positions (1 / Σw²)")
    fig.suptitle(f"Markowitz portfolios ({title})", x=0.01, ha="left", fontweight="bold")
    fig.subplots_adjust(wspace=0.4)
    save(fig, out / "10_turnover_diversification.png")

    # 11 per-rebalance gain of ML average over historical
    ml_r = mv[mv.source.isin(MODELS)].groupby("Date").period_return.mean()
    diff = (ml_r - hist.period_return) * 100
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(diff.index, diff.values, width=12, color=np.where(diff >= 0, COLOR["ML average"], INK_2))
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("percentage points")
    date_ticks(ax)
    ax.set_title(f"ML average minus historical, each 20-day period: ML ahead in {(diff > 0).mean():.0%} of periods "
                 f"({title})", loc="left")
    save(fig, out / "11_ml_minus_historical_by_period.png")

    # 12 long leg vs short leg (long-short only)
    if side == "longshort":
        legs = v["legs"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for ax, s in zip(axes, ["Historical", "ML average"]):
            if s == "ML average":
                lg = pd.DataFrame({k: legs[[(f"{m} MV", k) for m in MODELS]].mean(axis=1) for k in ("long", "short")})
                tot = rets["ML average MV"]
            else:
                lg, tot = legs[f"{s} MV"], rets[f"{s} MV"]
            ax.plot(lg.index, lg["long"].cumsum() * 100, color=SERIES[2], lw=1.8, label="long leg (top 10)")
            ax.plot(lg.index, lg["short"].cumsum() * 100, color=SERIES[1], lw=1.8, label="short leg (bottom 10)")
            ax.plot(tot.index, tot.cumsum() * 100, color=INK, lw=2.2, label="total (net of costs)")
            ax.axhline(0, color=INK_2, lw=0.8)
            ax.set_title(f"{s}: cumulative contribution (%)", loc="left")
        legend_below(axes[0], 3)
        date_ticks(*axes)
        fig.suptitle(f"Where the long-short return comes from, Markowitz ({title})", x=0.01, ha="left",
                     fontweight="bold")
        save(fig, out / "12_long_vs_short_leg.png")


def compare_figures(all_v):
    out = FIG / "compare"
    labels = [label(s, lb).replace(", ", "\n") for s, lb in VARIANTS]

    # A: Sharpe by variant, averaged over the 20 start days
    if all(v["runs"] is not None for v in all_v.values()):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
        for ax, wt in zip(axes, WEIGHTING):
            rows = []
            for key in VARIANTS:
                r = all_v[key]["runs"]
                r = r[r.weighting.isin([wt, "-"])]
                hist = r[r.source == "Historical"].Sharpe
                ml = r[r.source.isin(MODELS)].groupby("offset").Sharpe.mean()
                bench = r[r.source == "Benchmark"].Sharpe
                rows.append((hist.mean(), ml.mean(), bench.mean(), hist.std(), ml.std()))
            rows = np.array(rows)
            x = np.arange(len(VARIANTS))
            ax.bar(x - 0.27, rows[:, 0], 0.26, yerr=rows[:, 3], color=COLOR["Historical"], label="Historical μ",
                   error_kw={"ecolor": INK, "lw": 1})
            ax.bar(x, rows[:, 1], 0.26, yerr=rows[:, 4], color=COLOR["ML average"], label="ML μ (mean of 7 models)",
                   error_kw={"ecolor": INK, "lw": 1})
            ax.bar(x + 0.27, rows[:, 2], 0.26, color=INK, alpha=0.35, label="Benchmark")
            for xi, (h, m, b, *_ ) in zip(x, rows):
                for dx, val in ((-0.27, h), (0, m), (0.27, b)):
                    ax.text(xi + dx, max(val, 0), f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
            ax.set_xticks(x, labels)
            ax.axhline(0, color=INK_2, lw=0.8)
            ax.set_title(WEIGHTING[wt].capitalize(), loc="left")
            ax.grid(axis="x", visible=False)
        legend_below(axes[0], 3)
        fig.suptitle("Sharpe ratio, mean over the 20 start days (± 1 sd)", x=0.01, ha="left", fontweight="bold")
        save(fig, out / "A_sharpe_by_variant.png")

        # B: share of start days on which each model beats historical, and the Sharpe gap
        for col, fname, ttl, fmt, cmap, center in [
                ("beat", "B_share_of_start_days_beating_historical.png",
                 "Share of the 20 start days on which ML beats historical (Sharpe)", "{:.0%}", SEQUENTIAL, None),
                ("gap", "C_sharpe_gap_vs_historical.png",
                 "Sharpe of ML minus Sharpe of historical, mean over 20 start days", "{:+.2f}", DIVERGING, 0)]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
            for ax, wt in zip(axes, WEIGHTING):
                mat = []
                for key in VARIANTS:
                    r = all_v[key]["runs"]
                    sh = r[r.weighting == wt].pivot(index="offset", columns="source", values="Sharpe")
                    d = sh[MODELS].sub(sh["Historical"], axis=0)
                    mat.append((d > 0).mean() if col == "beat" else d.mean())
                mat = pd.DataFrame(mat, index=labels).T
                if center is None:
                    ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
                else:
                    lim = np.abs(mat.values).max()
                    ax.imshow(mat.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        ax.text(j, i, fmt.format(mat.iat[i, j]), ha="center", va="center", fontsize=9, color=INK)
                ax.set_xticks(range(len(labels)), labels, fontsize=8)
                ax.set_yticks(range(len(MODELS)), MODELS)
                ax.grid(False)
                ax.set_title(WEIGHTING[wt], loc="left")
            fig.suptitle(ttl, x=0.01, ha="left", fontweight="bold")
            save(fig, out / fname)

    # D: CAGR and max drawdown by variant (start day 0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, metric, ttl in [(axes[0], "CAGR", "CAGR (%)"), (axes[1], "max DD", "Maximum drawdown (%)")]:
        x = np.arange(len(VARIANTS))
        for dx, s in ((-0.27, "Historical"), (0, "ML average"), (0.27, "Benchmark")):
            vals = [stats(all_v[k]["rets"]["Benchmark" if s == "Benchmark" else f"{s} MV"])[metric] * 100
                    for k in VARIANTS]
            ax.bar(x + dx, vals, 0.26, color=COLOR[s], alpha=0.35 if s == "Benchmark" else 1, label=s)
            for xi, val in zip(x, vals):
                ax.text(xi + dx, val, f"{val:.0f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8,
                        color=INK_2)
        ax.set_xticks(x, labels)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_title(f"{ttl}, Markowitz", loc="left")
        ax.grid(axis="x", visible=False)
    legend_below(axes[0], 3)
    save(fig, out / "D_cagr_and_drawdown_by_variant.png")

    # E: equity curves of every variant on one page
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, key in zip(axes.flat, VARIANTS):
        rets = all_v[key]["rets"]
        for s in ["Historical", "ML average", "Benchmark"]:
            r = rets["Benchmark"] if s == "Benchmark" else rets[f"{s} MV"]
            ax.plot(r.index, (1 + r).cumprod(), label=s, **style(s))
        ax.axhline(1, color=INK_2, lw=0.6)
        ax.set_title(label(*key), loc="left")
    legend_below(axes[1, 0], 3)
    date_ticks(*axes.flat)
    fig.suptitle("Growth of 1, Markowitz: ML average vs historical μ in every variant", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, out / "E_equity_all_variants.png")

    # F: summary table as CSV/markdown for the README
    rows = []
    for key in VARIANTS:
        rets, runs = all_v[key]["rets"], all_v[key]["runs"]
        for s in ["Historical", "ML average", "Benchmark"]:
            for wt in (["-"] if s == "Benchmark" else list(WEIGHTING)):
                st = stats(rets["Benchmark" if s == "Benchmark" else f"{s} {wt}"])
                row = {"variant": label(*key), "mu source": s, "weighting": wt, **st}
                if runs is not None and s == "Historical":
                    row["Sharpe, mean of 20 start days"] = runs[(runs.source == s) & (runs.weighting == wt)].Sharpe.mean()
                elif runs is not None and s == "ML average":
                    row["Sharpe, mean of 20 start days"] = runs[runs.source.isin(MODELS) & (runs.weighting == wt)] \
                        .groupby("offset").Sharpe.mean().mean()
                rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(RES / "variant_summary.csv", index=False)
    show = t.copy()
    for c in ["CAGR", "vol", "max DD"]:
        show[c] = show[c].map("{:.1%}".format)
    for c in ["Sharpe", "Sharpe, mean of 20 start days"]:
        show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    (RES / "variant_summary.md").write_text("# All variants: historical vs ML average vs benchmark\n\n"
                                            + show.to_markdown(index=False))
    print(show.to_string(index=False))


all_v = {}
for side, lb in VARIANTS:
    if (RES / f"{side}_lb{lb}" / "daily_returns.parquet").exists():
        all_v[(side, lb)] = load(side, lb)
        variant_figures(side, lb, all_v[(side, lb)])
if len(all_v) == len(VARIANTS):
    compare_figures(all_v)
print(f"figures -> {FIG}")
