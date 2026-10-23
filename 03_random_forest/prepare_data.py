"""Random Forest — EDA, feature selection and model-ready dataset.

    .venv/bin/python 03_random_forest/prepare_data.py

Trees are invariant to monotone transforms and capture non-linearity and
interactions, so (unlike the linear models):
  * market-regime columns are kept: a split on bench_vol_20d can switch which
    stock features matter;
  * stock features are per-date ranks (removes drift in levels across years so
    a split learned in 2019 means the same thing in 2025);
  * selection is model-based: Boruta-style shadow features + permutation
    importance measured as loss of validation rank IC.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from common import config as C  # noqa: E402
from common.data import cs_rank, load_clean  # noqa: E402
from common.plotting import BLUE, INK_2, ORANGE, SERIES, plt, save  # noqa: E402
from common.prep import (add_targets, every_kth_date, ic_with_t, permutation_ic,  # noqa: E402
                         split_table, write_dataset, write_json)
from common.selection import daily_rank_ic, prune_redundant, quantile_spread  # noqa: E402

FIG = HERE / "figures"
OUT = HERE / "data" / "rf_dataset.parquet"

CORR_THRESHOLD = 0.90
RF_PARAMS = dict(n_estimators=300, max_depth=8, min_samples_leaf=200, max_features=0.33,
                 n_jobs=-1, random_state=C.SEED)
THIN = 5          # every 5th train date for fitting (20d targets overlap)
SHADOW_PCTL = 90  # real feature must beat the 90th pct of shadow importances


def build_matrix(df):
    stock = cs_rank(df, C.STOCK_CANDIDATES).fillna(0.0)       # NaN -> median rank
    mkt = df[C.MARKET].copy()
    med = mkt[df.split.eq("train")].median()
    mkt = mkt.fillna(med)                                        # only burn-in rows are NaN
    return pd.concat([stock, mkt], axis=1)


def main():
    df = add_targets(load_clean())
    ic = pd.read_csv(C.FEATURE_IC, index_col=0)
    corr = pd.read_csv(C.FEATURE_CORR, index_col=0)
    X = build_matrix(df)
    train = df.split.eq("train") & df.has_target
    val = df.split.eq("val") & df.has_target
    fit_mask = train & every_kth_date(df[train], THIN).reindex(df.index, fill_value=False)

    # ---- step 1: redundancy. Stock features on cross-sectional rank corr; market on daily time-series corr
    s_stock, cl_stock = prune_redundant(corr, ic.abs_t, CORR_THRESHOLD, candidates=C.STOCK_CANDIDATES)
    daily = df.loc[df.split.eq("train")].drop_duplicates("Date")[C.MARKET]
    mcorr = daily.corr(method="spearman")
    mscore = pd.Series(1.0, index=C.MARKET)          # no IC for market cols: keep first in list order
    mscore[:] = np.linspace(1, 0.5, len(C.MARKET))
    s_mkt, cl_mkt = prune_redundant(mcorr, mscore, CORR_THRESHOLD, candidates=C.MARKET)
    s1 = s_stock + s_mkt

    # ---- step 2: Boruta-style shadow test (one round) on train
    rng = np.random.default_rng(C.SEED)
    Xf = X.loc[fit_mask, s1]
    shadows = Xf.apply(lambda c: pd.Series(rng.permutation(c.to_numpy()), index=c.index))
    shadows.columns = [f"shadow_{c}" for c in s1]
    rf_b = RandomForestRegressor(**RF_PARAMS).fit(pd.concat([Xf, shadows], axis=1), df.loc[fit_mask, "y_win"])
    imp = pd.Series(rf_b.feature_importances_, index=list(s1) + list(shadows.columns))
    shadow_cut = np.percentile(imp[shadows.columns], SHADOW_PCTL)
    mdi = imp[s1].sort_values(ascending=False)
    pass_shadow = mdi[mdi > shadow_cut].index.tolist()

    # ---- step 3: permutation importance on validation (drop in rank IC, within-date shuffles)
    rf = RandomForestRegressor(**RF_PARAMS).fit(X.loc[fit_mask, s1], df.loc[fit_mask, "y_win"])
    vmask = val & every_kth_date(df[val], 2).reindex(df.index, fill_value=False)
    base_ic, perm = permutation_ic(rf.predict, X.loc[vmask, s1], df[vmask], repeats=3)
    pass_perm = perm[perm.ic_drop > 0].index.tolist()

    # Final: must beat the shadows on train AND not hurt validation when removed
    final = [f for f in s1 if f in pass_shadow and f in pass_perm]
    # keep regime columns that passed the shadow test even if their val drop is ~0:
    # they only matter through interactions, which permutation within a date cannot break
    final += [f for f in s_mkt if f in pass_shadow and f not in final]

    comp = []
    for name, feats in {"after redundancy": s1, "selected": final}.items():
        m = RandomForestRegressor(**RF_PARAMS).fit(X.loc[fit_mask, feats], df.loc[fit_mask, "y_win"])
        v, t = ic_with_t(m.predict(X.loc[val, feats]), df[val])
        comp.append({"feature set": name, "n": len(feats), "val IC": v, "val IC NW t": t})
    comp = pd.DataFrame(comp)

    # ---- EDA: non-linearity (deciles) and regime dependence
    top_nl = [f for f in perm.index if f in s_stock][:6]
    tr = df[train]
    dec = {}
    for f in top_nl:
        dec[f] = quantile_spread(tr, f, target="y_raw", q=10) * 100

    regime = pd.qcut(df.loc[train, "bench_vol_20d"], 3, labels=["low vol", "mid vol", "high vol"])
    reg_ic = {}
    for lab in regime.cat.categories:
        sub = df.loc[train][regime == lab]
        reg_ic[lab] = daily_rank_ic(sub, top_nl).mean()
    reg_ic = pd.DataFrame(reg_ic)

    # ---- write dataset
    full = df[["Date", "Symbol", "split", "has_target", "tradable", "y_raw", "y_win", "y_rank", "y_class"]].join(X[final])
    write_dataset(full, final, OUT)
    write_json({"model": "random_forest", "features": final,
                "stock_features": [f for f in final if f in C.STOCK_CANDIDATES],
                "market_features": [f for f in final if f in C.MARKET],
                "transform": "stock: per-date percentile rank centred to [-0.5, 0.5], NaN -> 0; "
                             "market: raw, NaN -> train median",
                "target_default": "y_win", "eval_target": "y_raw",
                "selection_rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
                "fit_on_every_kth_train_date": THIN},
               HERE / "selected_features.json")

    # ---- figures
    fig, ax = plt.subplots(figsize=(8, 10))
    m2 = mdi.sort_values()
    ax.barh(m2.index, m2.values, color=np.where(m2.values > shadow_cut, ORANGE, BLUE))
    ax.axvline(shadow_cut, color=INK_2, ls="--", lw=0.8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set(title="Boruta-style test: MDI importance vs shuffled shadows (train)",
           xlabel=f"impurity importance (dashed = {SHADOW_PCTL}th pct of shadows)")
    save(fig, FIG / "01_shadow_importance.png")

    fig, ax = plt.subplots(figsize=(8, 10))
    p2 = perm.sort_values("ic_drop")
    ax.barh(p2.index, p2.ic_drop, xerr=p2.ic_drop_sd, color=np.where(p2.ic_drop > 0, ORANGE, BLUE),
            error_kw={"ecolor": INK_2, "lw": 0.8})
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set(title=f"Permutation importance on validation (base IC {base_ic:.3f})",
           xlabel="drop in mean daily rank IC when shuffled within date")
    save(fig, FIG / "02_permutation_importance.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for k, (f, s) in enumerate(dec.items()):
        ax.plot(s.index + 1, s.values, marker="o", ms=4, color=SERIES[k], label=f)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(range(1, 11))
    ax.set(title="Non-linearity: mean 20d excess return by feature decile (train)",
           xlabel="decile on each date", ylabel="%")
    ax.legend(fontsize=8, ncol=2)
    save(fig, FIG / "03_decile_shapes.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.27
    for k, c in enumerate(reg_ic.columns):
        ax.bar(np.arange(len(reg_ic)) + (k - 1) * w, reg_ic[c], width=w, color=SERIES[k], label=c)
    ax.set_xticks(range(len(reg_ic)), reg_ic.index, rotation=20, fontsize=8)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(title="Feature IC by market-volatility regime (train)", ylabel="mean daily IC")
    ax.legend()
    save(fig, FIG / "04_regime_ic.png")

    # ---- report
    t = pd.DataFrame({"MDI": mdi, "beats_shadow": mdi > shadow_cut}).join(perm).join(
        ic[["ic_mean", "pdf_tier"]])
    t["selected"] = t.index.isin(final)
    t = t.sort_values("ic_drop", ascending=False)
    md = [
        "# 03 — Random Forest: EDA & feature selection",
        "",
        "Prerequisite: `00_data_audit/REPORT.md`.",
        "",
        "## Why RF is prepared differently",
        "- Trees split on order, not scale, so no winsorizing/z-scoring is needed. Stock features are still converted to "
        "**per-date percentile ranks**, so a split threshold means the same thing in every year (raw volatility or volume levels drift).",
        "- **Market-regime columns are kept** (raw): trees can interact them with stock features. The regime chart below "
        "shows several features whose IC changes size or sign with market volatility.",
        "- Correlated features split importance between them and make importances unreadable → near-duplicates removed "
        f"(|corr| ≥ {CORR_THRESHOLD}).",
        f"- Fitting uses every {THIN}th train date: consecutive 20-day targets overlap 95%, so the extra rows add almost "
        "no information but inflate leaf purity (overfitting).",
        "",
        "## Selection",
        "",
        f"1. Redundancy: stock {len(C.STOCK_CANDIDATES)} → {len(s_stock)}, market {len(C.MARKET)} → {len(s_mkt)} "
        f"(dropped market: {', '.join(f'`{f}`' for f in C.MARKET if f not in s_mkt) or 'none'}).",
        f"2. Boruta-style shadow test: {len(pass_shadow)} of {len(s1)} beat the {SHADOW_PCTL}th percentile of shuffled copies.",
        f"3. Permutation importance on validation (shuffle within date, 3 repeats): {len(pass_perm)} have a positive IC drop.",
        f"4. **Selected = passes 2 and 3** (market columns need only pass 2, since within-date shuffling cannot "
        f"measure a date-constant feature) → **{len(final)} features**: " + ", ".join(f"`{f}`" for f in final),
        "",
        "![shadow](figures/01_shadow_importance.png)",
        "![perm](figures/02_permutation_importance.png)",
        "",
        t.to_markdown(floatfmt=".4g"),
        "",
        "## Non-linearity and regimes (why a tree model is worth trying)",
        "",
        "![deciles](figures/03_decile_shapes.png)",
        "![regimes](figures/04_regime_ic.png)",
        "",
        reg_ic.to_markdown(floatfmt=".4f"),
        "",
        "## Sanity check on validation",
        "",
        comp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Caveat: validation was also used for the permutation step, so the `selected` row is mildly optimistic. "
        "The test period stays untouched.",
        "",
        "## Model-ready dataset",
        "",
        f"`data/rf_dataset.parquet` — {len(full):,} rows, {len(final)} features (float32, no NaN), targets `y_raw, y_win, y_rank, y_class`.",
        "",
        split_table(full).to_markdown(),
        "",
        "Use rows with `split == 'train'` and `has_target` (thin to every 5th date, or use `max_samples`), "
        "tune `max_depth`, `min_samples_leaf`, `max_features` on `val`, evaluate with `y_raw`.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(md))
    print("selected", final)
    print(comp)


if __name__ == "__main__":
    main()
