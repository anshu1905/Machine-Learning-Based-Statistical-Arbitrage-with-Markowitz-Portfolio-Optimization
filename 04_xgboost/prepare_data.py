"""XGBoost — EDA, feature selection and model-ready dataset.

    .venv/bin/python 04_xgboost/prepare_data.py

Gradient boosting differs from RF in two ways that matter for preparation:
  * it handles NaN natively (learns a default branch per split) -> missing
    values are KEPT, and we check whether missingness itself is informative;
  * it is more prone to chasing noise with many weak features -> selection uses
    SHAP contributions on validation plus validation permutation importance,
    with early stopping on validation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import xgboost as xgb  # noqa: E402

from common import config as C  # noqa: E402
from common.data import cs_rank, load_clean  # noqa: E402
from common.plotting import BLUE, INK_2, ORANGE, plt, save  # noqa: E402
from common.prep import (add_targets, every_kth_date, ic_with_t, permutation_ic,  # noqa: E402
                         split_table, write_dataset, write_json)
from common.selection import prune_redundant  # noqa: E402

FIG = HERE / "figures"
OUT = HERE / "data" / "xgb_dataset.parquet"

CORR_THRESHOLD = 0.95
SHAP_CUM = 0.95
THIN = 5
XGB_PARAMS = dict(n_estimators=2000, learning_rate=0.03, max_depth=4, min_child_weight=200,
                  subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
                  early_stopping_rounds=100, n_jobs=-1, random_state=C.SEED)


def build_matrix(df):
    stock = cs_rank(df, C.STOCK_CANDIDATES)        # NaN kept on purpose
    return pd.concat([stock, df[C.MARKET]], axis=1)


def fit(X, y, Xv, yv):
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    return m


def main():
    df = add_targets(load_clean())
    ic = pd.read_csv(C.FEATURE_IC, index_col=0)
    corr = pd.read_csv(C.FEATURE_CORR, index_col=0)
    X = build_matrix(df)
    train = df.split.eq("train") & df.has_target
    val = df.split.eq("val") & df.has_target
    fm = train & every_kth_date(df[train], THIN).reindex(df.index, fill_value=False)
    vm = val & every_kth_date(df[val], 2).reindex(df.index, fill_value=False)

    # ---- missing-value EDA: is missingness informative?
    miss = []
    tr = df[train]
    for f in C.STOCK_CANDIDATES + C.MARKET:
        isna = tr[f].isna()
        if isna.sum() >= 200:
            miss.append({"feature": f, "na_share_train": isna.mean(),
                         "na_share_val": df.loc[val, f].isna().mean(),
                         "mean_y_when_na": tr.loc[isna, "y_raw"].mean(),
                         "mean_y_when_present": tr.loc[~isna, "y_raw"].mean()})
    miss = pd.DataFrame(miss).set_index("feature") if miss else pd.DataFrame()

    # ---- step 1: exact / near-exact duplicates only
    s_stock, _ = prune_redundant(corr, ic.abs_t, CORR_THRESHOLD, candidates=C.STOCK_CANDIDATES)
    mcorr = df.loc[df.split.eq("train")].drop_duplicates("Date")[C.MARKET].corr(method="spearman")
    s_mkt, _ = prune_redundant(mcorr, pd.Series(np.linspace(1, .5, len(C.MARKET)), index=C.MARKET),
                               CORR_THRESHOLD, candidates=C.MARKET)
    s1 = s_stock + s_mkt

    # ---- step 2: fit with early stopping, SHAP on validation
    m = fit(X.loc[fm, s1], df.loc[fm, "y_win"], X.loc[vm, s1], df.loc[vm, "y_win"])
    contrib = m.get_booster().predict(xgb.DMatrix(X.loc[vm, s1]), pred_contribs=True)[:, :-1]
    shap = pd.Series(np.abs(contrib).mean(0), index=s1).sort_values(ascending=False)
    cum = shap.cumsum() / shap.sum()
    in_shap = cum[cum.shift(fill_value=0) < SHAP_CUM].index.tolist()
    gain = pd.Series(m.get_booster().get_score(importance_type="total_gain")).reindex(s1).fillna(0)

    # ---- step 3: permutation importance on validation
    base_ic, perm = permutation_ic(m.predict, X.loc[vm, s1], df[vm], repeats=3)
    hurts = perm[perm.ic_drop < -perm.ic_drop_sd].index.tolist()   # removing it clearly helps

    final = [f for f in s1 if f in in_shap and f not in hurts]

    comp = []
    for name, feats in {"all 58 (stock + market)": C.STOCK_CANDIDATES + C.MARKET,
                        "after duplicates": s1, "selected": final}.items():
        mm = fit(X.loc[fm, feats], df.loc[fm, "y_win"], X.loc[vm, feats], df.loc[vm, "y_win"])
        v, t = ic_with_t(mm.predict(X.loc[val, feats]), df[val])
        comp.append({"feature set": name, "n": len(feats), "trees (early stop)": mm.best_iteration + 1,
                     "val IC": v, "val IC NW t": t})
    comp = pd.DataFrame(comp)

    # ---- write dataset
    full = df[["Date", "Symbol", "split", "has_target", "tradable", "y_raw", "y_win", "y_rank", "y_class"]].join(X[final])
    write_dataset(full, final, OUT)
    write_json({"model": "xgboost", "features": final,
                "stock_features": [f for f in final if f in C.STOCK_CANDIDATES],
                "market_features": [f for f in final if f in C.MARKET],
                "transform": "stock: per-date percentile rank centred to [-0.5, 0.5], NaN KEPT; market: raw, NaN KEPT",
                "target_default": "y_win", "eval_target": "y_raw",
                "selection_xgb_params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
                "selection_best_iteration": int(m.best_iteration + 1),
                "fit_on_every_kth_train_date": THIN},
               HERE / "selected_features.json")

    # ---- figures
    fig, ax = plt.subplots(figsize=(8, 10))
    s2 = shap.sort_values()
    ax.barh(s2.index, s2.values, color=np.where(s2.index.isin(final), ORANGE, BLUE))
    ax.tick_params(axis="y", labelsize=8)
    ax.set(title="Mean |SHAP| on validation (orange = selected)", xlabel="mean |contribution| to prediction")
    save(fig, FIG / "01_shap_importance.png")

    fig, ax = plt.subplots(figsize=(8, 10))
    p2 = perm.sort_values("ic_drop")
    ax.barh(p2.index, p2.ic_drop, xerr=p2.ic_drop_sd, color=np.where(p2.ic_drop > 0, ORANGE, BLUE),
            error_kw={"ecolor": INK_2, "lw": 0.8})
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set(title=f"Permutation importance on validation (base IC {base_ic:.3f})",
           xlabel="drop in mean daily rank IC when shuffled within date")
    save(fig, FIG / "02_permutation_importance.png")

    top4 = shap.index[:4]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    xv = X.loc[vm, s1]
    for ax, f in zip(axes, top4):
        j = s1.index(f)
        idx = np.random.default_rng(0).choice(len(xv), size=min(6000, len(xv)), replace=False)
        ax.scatter(xv[f].to_numpy()[idx], contrib[idx, j], s=3, alpha=0.3, color=BLUE, lw=0)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_title(f, fontsize=9)
    axes[0].set_ylabel("SHAP contribution")
    fig.suptitle("Learned shape of the top-4 features (validation)", fontweight="bold")
    save(fig, FIG / "03_shap_dependence.png")

    if len(miss):
        fig, ax = plt.subplots(figsize=(8, 0.6 + 0.3 * len(miss)))
        mm2 = miss.sort_values("na_share_train")
        ax.barh(mm2.index, mm2.na_share_train * 100, color=BLUE)
        ax.tick_params(axis="y", labelsize=8)
        ax.set(title="Missing share in train (kept as NaN for XGBoost)", xlabel="% of train rows")
        save(fig, FIG / "04_missing_train.png")

    # ---- report
    t = pd.DataFrame({"mean_abs_shap": shap, "cum_share": cum, "total_gain": gain}).join(perm).join(
        ic[["ic_mean", "pdf_tier"]])
    t["selected"] = t.index.isin(final)
    md = [
        "# 04 — XGBoost: EDA & feature selection",
        "",
        "Prerequisite: `00_data_audit/REPORT.md`.",
        "",
        "## Why XGBoost is prepared differently",
        "- **Missing values are kept.** XGBoost learns a default direction for NaN at every split, so imputing would "
        "throw away the information that a stock is newly listed / has short history. The table below checks whether "
        "missingness relates to the target.",
        "- Stock features are per-date ranks (stable split thresholds across years); market-regime columns are kept raw.",
        f"- Only near-exact duplicates are removed first (|corr| ≥ {CORR_THRESHOLD}); boosting then reveals usefulness through "
        "SHAP and validation permutation importance. Early stopping on validation guards the selection fit.",
        "",
        "## Missingness",
        "",
        (miss.to_markdown(floatfmt=".4f") if len(miss) else "No feature has ≥ 200 missing train rows."),
        "",
        "![missing](figures/04_missing_train.png)" if len(miss) else "",
        "",
        "## Selection",
        "",
        f"1. Duplicates removed → {len(s1)} ({len(s_stock)} stock + {len(s_mkt)} market).",
        f"2. XGBoost (depth {XGB_PARAMS['max_depth']}, lr {XGB_PARAMS['learning_rate']}, early-stopped at "
        f"{m.best_iteration + 1} trees) on every {THIN}th train date. Features that make up {SHAP_CUM:.0%} of total "
        f"mean |SHAP| on validation: {len(in_shap)}.",
        f"3. Removed features whose within-date shuffle *raises* validation IC by more than one s.d.: "
        f"{', '.join(f'`{f}`' for f in hurts) or 'none'}.",
        f"→ **{len(final)} features**: " + ", ".join(f"`{f}`" for f in final),
        "",
        "![shap](figures/01_shap_importance.png)",
        "![perm](figures/02_permutation_importance.png)",
        "![dependence](figures/03_shap_dependence.png)",
        "",
        t.to_markdown(floatfmt=".4g"),
        "",
        "## Sanity check on validation",
        "",
        comp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Caveat: validation is used both for early stopping and for this check, so these numbers are mildly optimistic. "
        "The test period stays untouched.",
        "",
        "## Model-ready dataset",
        "",
        f"`data/xgb_dataset.parquet` — {len(full):,} rows, {len(final)} features (float32, **NaN retained**), targets "
        "`y_raw, y_win, y_rank, y_class`.",
        "",
        split_table(full).to_markdown(),
        "",
        "Use rows with `split == 'train'` and `has_target`; early-stop on `val`; evaluate with `y_raw`. "
        "`y_rank` with `objective='reg:squarederror'` or `rank:pairwise` grouped by Date are both worth testing.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(md))
    print("selected", final)
    print(comp)


if __name__ == "__main__":
    main()
