"""Shared EDA + data audit. Run once before any model folder.

    .venv/bin/python 00_data_audit/run_audit.py

Writes
    data/panel_clean.parquet      audited panel: raw cols + extra features + split + flags
    data/feature_ic_train.csv     univariate rank-IC of every candidate (TRAIN only)
    data/feature_corr_train.csv   cross-sectional rank correlation (TRAIN only)
    00_data_audit/figures/*.png, 00_data_audit/REPORT.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import config as C  # noqa: E402
from common.data import assign_splits  # noqa: E402
from common.features_extra import add_extra_features  # noqa: E402
from common.plotting import BLUE, DIVERGING, INK_2, ORANGE, plt, save  # noqa: E402
from common.selection import cs_rank_corr, daily_rank_ic, ic_summary, quantile_spread  # noqa: E402

FIG = HERE / "figures"
SPLIT_SHADE = {"train": "#e8f0fb", "val": "#fdeee6", "test": "#e6f5ee"}


def build_raw():
    from python_calamine import CalamineWorkbook
    wb = CalamineWorkbook.from_path(C.RAW_XLSX)
    frames = []
    for name in wb.sheet_names:
        rows = wb.get_sheet_by_name(name).to_python()
        frames.append(pd.DataFrame(rows[1:], columns=rows[0]))
    df = pd.concat(frames, ignore_index=True).replace("", np.nan)
    for c in df.columns.drop(["Date", "Symbol"]):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"])
    C.DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(C.PANEL_RAW, index=False)
    return df


def shade_splits(ax, df):
    for s, col in SPLIT_SHADE.items():
        d = df.loc[df.split == s, "Date"]
        ax.axvspan(d.min(), d.max(), color=col, zorder=0, lw=0)
        ax.text(d.min() + (d.max() - d.min()) / 2, 0.97, s, transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=INK_2, fontsize=9)


def main():
    df = pd.read_parquet(C.PANEL_RAW) if C.PANEL_RAW.exists() else build_raw()
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    log = []

    # ------------------------------------------------------------------ checks
    checks = {
        "duplicate (Symbol, Date) keys": int(df.duplicated(["Symbol", "Date"]).sum()),
        "max |target - (fwd_ret_20d - bench_fwd_ret)|":
            float(np.nanmax(np.abs(df[C.TARGET] - (df.fwd_ret_20d - df.bench_fwd_ret)))),
        "max |fwd_ret_20d(t) - mom_20d(t+20)|":
            float(np.nanmax(np.abs(df.fwd_ret_20d - df.groupby("Symbol").mom_20d.shift(-20)))),
        "max |candle_body - intraday_return|":
            float(np.nanmax(np.abs(df.candle_body - df.intraday_return))),
        "max |rel_mom_20d - (mom_20d - bench_ret_20d)|":
            float(np.nanmax(np.abs(df.rel_mom_20d - (df.mom_20d - df.bench_ret_20d)))),
        "max per-date unique values of r_b (market col)": int(df.groupby("Date").r_b.nunique().max()),
        "rows with Volume == 0": int((df.Volume == 0).sum()),
        "rows with High == Low (locked / no range)": int((df.High == df.Low).sum()),
        "rows with |ret_1d| > 35% (corporate action or crash)": int((df.ret_1d.abs() > 0.35).sum()),
        "VWAP missing share": float(df.VWAP.isna().mean()),
        "weekend sessions (Muhurat / Budget days)": int(df.Date.dt.dayofweek.ge(5).groupby(df.Date).any().sum()),
    }

    # ------------------------------------------------------- extra features
    df = add_extra_features(df)
    df["split"] = assign_splits(df["Date"])
    df["has_target"] = df[C.TARGET].notna()
    df["tradable"] = (df.Volume > 0) & (df.High > df.Low)
    df["big_jump"] = df.ret_1d.abs() > 0.35
    df = df.drop(columns=list(C.DUPLICATES))

    keep_cols = (C.ID_COLS + ["split", "has_target", "tradable", "big_jump"] + C.RAW_OHLCV
                 + C.MARKET + C.STOCK_CANDIDATES + C.PRE_RANKED + C.LEAKAGE)
    df = df[keep_cols]
    df.to_parquet(C.PANEL_CLEAN, index=False)

    split_counts = df.groupby("split").agg(rows=("Date", "size"), dates=("Date", "nunique"),
                                           first=("Date", "min"), last=("Date", "max"))

    # ------------------------------------------------------- universe / splits
    fig, ax = plt.subplots(figsize=(10, 3.2))
    n = df.groupby("Date").Symbol.size()
    shade_splits(ax, df)
    ax.plot(n.index, n.values, color=BLUE)
    ax.set(title="Stocks in the universe per trading day", ylabel="stocks")
    save(fig, FIG / "01_universe_and_splits.png")

    # ------------------------------------------------------- missingness
    feats = C.STOCK_CANDIDATES + C.MARKET
    post = df[df.split != "burn_in"]
    na = pd.DataFrame({"all_rows": df[feats].isna().mean(), "after_burn_in": post[feats].isna().mean()})
    na = na.sort_values("all_rows")
    fig, ax = plt.subplots(figsize=(8, 11))
    y = np.arange(len(na))
    ax.barh(y + 0.2, na.all_rows * 100, height=0.4, color=BLUE, label="all rows (incl. 2018)")
    ax.barh(y - 0.2, na.after_burn_in * 100, height=0.4, color=ORANGE, label="from 2019")
    ax.set_yticks(y, na.index, fontsize=8)
    ax.set(title="Missing values by feature", xlabel="% missing")
    ax.legend(loc="lower right")
    save(fig, FIG / "02_missingness.png")

    # ------------------------------------------------------- target
    t = df.loc[df.has_target & (df.split != "burn_in")]
    disp = t.groupby("Date")[C.TARGET].agg(["mean", "std", lambda x: x.quantile(.9) - x.quantile(.1)])
    disp.columns = ["mean", "std", "p90_p10"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].hist(t[C.TARGET].clip(-0.6, 0.6), bins=120, color=BLUE)
    axes[0].set_yscale("log")
    axes[0].set(title="20d excess return (clipped ±60%)", xlabel="excess return", ylabel="rows (log)")
    shade_splits(axes[1], df)
    axes[1].plot(disp.index, disp.p90_p10, color=BLUE, lw=1.4, label="P90 − P10 across stocks")
    axes[1].plot(disp.index, disp["mean"], color=ORANGE, lw=1.4, label="cross-sectional mean")
    axes[1].axhline(0, color=INK_2, lw=0.8)
    axes[1].set(title="Cross-sectional dispersion of the target")
    axes[1].legend(loc="upper left")
    save(fig, FIG / "03_target.png")

    tgt_stats = t[C.TARGET].describe(percentiles=[.01, .05, .5, .95, .99])
    tgt_stats["skew"], tgt_stats["kurtosis"] = t[C.TARGET].skew(), t[C.TARGET].kurt()

    # ------------------------------------------------------- benchmark regime
    b = df.drop_duplicates("Date").set_index("Date")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    shade_splits(ax, df)
    ax.plot(b.index, (1 + b.r_b.fillna(0)).cumprod(), color=BLUE)
    ax.set(title="Benchmark growth of ₹1 (regimes covered by each split)", ylabel="₹")
    save(fig, FIG / "04_benchmark.png")

    # ------------------------------------------------------- distributions
    train = df[(df.split == "train") & df.has_target]
    shape = pd.DataFrame({"skew": train[C.STOCK_CANDIDATES].skew(),
                          "kurtosis": train[C.STOCK_CANDIDATES].kurt()})
    s = shape["kurtosis"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(s.index, np.log10(s.clip(lower=0.01) + 1), color=np.where(s > 10, ORANGE, BLUE))
    ax.axvline(np.log10(11), color=INK_2, lw=0.8, ls="--")
    ax.tick_params(axis="y", labelsize=8)
    ax.set(title="Excess kurtosis by feature (train)", xlabel="log10(1 + kurtosis); dashed = 10")
    save(fig, FIG / "05_feature_kurtosis.png")

    # ------------------------------------------------------- univariate IC
    ic = daily_rank_ic(train, C.STOCK_CANDIDATES)
    summ = ic_summary(ic)
    summ["pdf_tier"] = pd.Series(C.PDF_TIER)
    summ["na_after_burn_in"] = na.after_burn_in
    summ = summ.join(shape)
    summ.index.name = "feature"
    summ.to_csv(C.FEATURE_IC)

    s = summ.sort_values("ic_mean")
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.barh(s.index, s.ic_mean, color=np.where(s.ic_mean > 0, ORANGE, BLUE))
    for yv, (m, tt) in enumerate(zip(s.ic_mean, s.ic_tstat_nw)):
        ax.text(m + (0.001 if m > 0 else -0.001), yv, f"t={tt:.1f}", va="center",
                ha="left" if m > 0 else "right", fontsize=7, color=INK_2)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.margins(x=0.12)
    ax.tick_params(axis="y", labelsize=8)
    ax.set(title="Mean daily rank IC vs 20d excess return (train 2019–22)",
           xlabel="mean Spearman IC (labels: Newey–West t-stat)")
    save(fig, FIG / "06_ic_train.png")

    top = summ.index[:25]
    by_year = ic[top].groupby(ic.index.year).mean().T
    fig, ax = plt.subplots(figsize=(6, 9))
    lim = np.nanmax(np.abs(by_year.values))
    im = ax.imshow(by_year.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(by_year.shape[1]), by_year.columns)
    ax.set_yticks(range(len(top)), top, fontsize=8)
    for i in range(by_year.shape[0]):
        for j in range(by_year.shape[1]):
            ax.text(j, i, f"{by_year.iat[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.5, label="mean IC")
    ax.set_title("IC by year — top 25 features by |t|")
    save(fig, FIG / "07_ic_by_year.png")

    # ------------------------------------------------------- redundancy
    corr = cs_rank_corr(train, C.STOCK_CANDIDATES)
    corr.to_csv(C.FEATURE_CORR)
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    d = 1 - corr.abs().to_numpy()
    np.fill_diagonal(d, 0)
    order = corr.index[leaves_list(linkage(squareform(d, checks=False), "average"))]
    cc = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(12, 10.5))
    im = ax.imshow(cc.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="rank correlation")
    ax.set_title("Cross-sectional rank correlation of candidates (train), clustered")
    save(fig, FIG / "08_feature_correlation.png")

    pairs = (corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
             .rename("rho").reset_index().assign(abs_rho=lambda x: x.rho.abs())
             .sort_values("abs_rho", ascending=False))
    hi_pairs = pairs[pairs.abs_rho >= 0.85]

    # ------------------------------------------------------- quintile spreads
    top8 = summ.index[:8]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharey=True)
    for ax, f in zip(axes.ravel(), top8):
        qs = quantile_spread(train, f) * 100
        ax.bar(qs.index + 1, qs.values, color=np.where(qs.values > 0, ORANGE, BLUE), width=0.7)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_title(f, fontsize=10)
        ax.set_xticks(range(1, 6))
    fig.supxlabel("feature quintile on each date (1 = lowest)")
    fig.supylabel("mean 20d excess return (%)")
    fig.suptitle("Quintile spreads of the 8 strongest features (train)", fontweight="bold")
    save(fig, FIG / "09_quintile_spreads.png")

    # ------------------------------------------------------- report
    fmt = lambda x: f"{x:.4g}" if isinstance(x, float) else str(x)  # noqa: E731
    lines = [
        "# 00 — Shared data audit and EDA",
        "",
        "Source: `features_by_stock.xlsx` (238 sheets, one per stock; Bloomberg OHLCV + engineered features).",
        f"Panel: **{len(df):,} rows**, **{df.Symbol.nunique()} stocks**, "
        f"{df.Date.min():%Y-%m-%d} → {df.Date.max():%Y-%m-%d}, {df.Date.nunique():,} trading days.",
        "",
        "## Integrity checks",
        "",
        "| check | value |", "|---|---|",
        *[f"| {k} | {fmt(v)} |" for k, v in checks.items()],
        "",
        "Conclusions:",
        "- `target_excess_ret_20d = fwd_ret_20d − bench_fwd_ret` exactly, and `fwd_ret_20d(t) = mom_20d(t+20)`: "
        "the target is the **simple 20-trading-day forward return in excess of the benchmark**, measured close t → close t+20.",
        "- `fwd_ret_20d`, `bench_fwd_ret`, `target_*` are future information → **excluded from every feature set**.",
        "- `candle_body` is an exact copy of `intraday_return` → dropped.",
        "- `rel_mom_k = mom_k − bench_ret_k`: inside one date they differ by a constant, so for a cross-sectional "
        "ranking model they are the same feature (rank correlation = 1).",
        "- The 8 `bench_*`/`r_b` columns are identical across stocks on a date (market regime). They cannot rank "
        "stocks by themselves; trees/RNNs may use them as regime conditioners, linear models drop them.",
        "- `rank_*` columns are per-date ranks of columns we already have → superseded by our own cross-sectional transforms.",
        "- VWAP is missing for all of 2018–2019 → not used.",
        "- A handful of |daily return| > 35% moves are unadjusted demergers (e.g. Siemens 2025-04-07, Tata Motors "
        "2025-10-14) or genuine crashes (Yes Bank 2020). They contaminate momentum features and targets for those names, "
        "which is why every model winsorizes features cross-sectionally (1/99%) and trains on a winsorized target.",
        "- Rows with zero volume or High = Low are flagged `tradable = False` (excluded from the selectable universe in the backtest).",
        "- Survivorship: the universe is today's 238 names back-filled to 2018 (8 names list later). Results will be "
        "optimistic vs a point-in-time index membership.",
        "",
        "## Splits (shared by all models)",
        "",
        f"Train {C.TRAIN_START} → {C.TRAIN_END}; validation {C.VAL_START} → {C.VAL_END}; test {C.TEST_START} → end. "
        f"2018 is burn-in (1-year lookbacks). The {C.HORIZON} trading days before validation and before test are "
        "**purged** so no target window crosses a boundary. The last 20 days have no target (live prediction rows).",
        "",
        split_counts.to_markdown(),
        "",
        "![universe](figures/01_universe_and_splits.png)",
        "![benchmark](figures/04_benchmark.png)",
        "",
        "## Extra features added from `statarb_features_ranked.pdf`",
        "",
        "The file did not contain several single-stock Tier-2/3 features from the PDF; these were computed causally from OHLCV: "
        + ", ".join(f"`{f}`" for f in C.STOCK_EXTRA) + ". "
        "Pair-trading features (spread, hedge ratio, cointegration, half-life, OU) are not applicable to this "
        "cross-sectional stock-ranking design; the factor-residual idea is already represented by `resid_ret_60d`.",
        "",
        "## Missing values",
        "",
        "Missingness is almost entirely the lookback warm-up (up to 252 days) plus late listings. After the 2018 burn-in "
        f"the worst feature is `{na.after_burn_in.idxmax()}` at {na.after_burn_in.max():.1%}.",
        "",
        "![missing](figures/02_missingness.png)",
        "",
        "## Target",
        "",
        tgt_stats.to_frame("target_excess_ret_20d").T.to_markdown(floatfmt=".4f"),
        "",
        "Heavy tails (kurtosis above) → models are trained on a cross-sectionally winsorized copy; the raw target is kept for evaluation.",
        "Targets overlap 19 of 20 days between consecutive dates, so naive t-stats are inflated; all IC t-stats here use Newey–West with 20 lags.",
        "",
        "![target](figures/03_target.png)",
        "",
        "## Feature distributions",
        "",
        f"{(shape['kurtosis'] > 10).sum()} of {len(shape)} candidates have excess kurtosis > 10 (orange) → cross-sectional winsorization / ranking is required.",
        "",
        "![kurtosis](figures/05_feature_kurtosis.png)",
        "",
        "## Univariate predictive power (train only)",
        "",
        "Daily Spearman rank IC between each feature and the 20d excess return, averaged over 2019–2022.",
        "",
        summ[["ic_mean", "ic_tstat_nw", "icir_ann", "hit_rate", "year_sign_stability", "pdf_tier",
              "na_after_burn_in"]].to_markdown(floatfmt=".4f"),
        "",
        "![ic](figures/06_ic_train.png)",
        "![ic year](figures/07_ic_by_year.png)",
        "![quintiles](figures/09_quintile_spreads.png)",
        "",
        "## Redundancy",
        "",
        f"{len(hi_pairs)} pairs have |cross-sectional rank correlation| ≥ 0.85:",
        "",
        hi_pairs.head(40)[["level_0", "level_1", "rho"]].rename(
            columns={"level_0": "feature A", "level_1": "feature B"}).to_markdown(index=False, floatfmt=".3f"),
        "",
        "![corr](figures/08_feature_correlation.png)",
        "",
        "Model-specific selection (each folder's REPORT.md) starts from this IC table and this correlation matrix.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print(split_counts)
    print(summ.head(15)[["ic_mean", "ic_tstat_nw"]])


if __name__ == "__main__":
    main()
