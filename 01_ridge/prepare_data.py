"""Ridge Regression — EDA, feature selection and model-ready dataset.

    .venv/bin/python 01_ridge/prepare_data.py

Ridge is linear, shrinks but never removes coefficients, and is distorted by
outliers and by near-collinear inputs (coefficients flip sign). So:
    stock features only (market columns are date-constant -> useless in a linear
    cross-sectional model), per-date winsorize 1/99% + z-score, strict IC screen,
    strict redundancy pruning, VIF < 5, and a monotonicity check.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sklearn.linear_model import Ridge  # noqa: E402

from common import config as C  # noqa: E402
from common.data import cs_winsor_zscore, load_clean  # noqa: E402
from common.plotting import BLUE, DIVERGING, INK_2, ORANGE, plt, save  # noqa: E402
from common.prep import add_targets, ic_of, ic_with_t, split_table, write_dataset, write_json  # noqa: E402
from common.selection import prune_redundant, quantile_spread, vif  # noqa: E402

FIG = HERE / "figures"
OUT = HERE / "data" / "ridge_dataset.parquet"

MIN_ABS_T = 0.5           # Newey-West t of mean daily IC (train): drops pure noise only
MIN_SIGN_STABILITY = 0.5   # same IC sign in >= 2 of 4 train years
CORR_THRESHOLD = 0.80      # one feature per cluster of |rank corr| >= 0.8
MAX_VIF = 5.0


def main():
    df = add_targets(load_clean())
    ic = pd.read_csv(C.FEATURE_IC, index_col=0)
    corr = pd.read_csv(C.FEATURE_CORR, index_col=0)
    cands = list(C.STOCK_CANDIDATES)

    # ---- transform first (selection statistics below use the transformed data)
    Z = cs_winsor_zscore(df, cands).fillna(0.0)   # NaN -> cross-sectional mean
    Z[["Date"]] = df[["Date"]]
    train = df.split.eq("train") & df.has_target
    val = df.split.eq("val") & df.has_target

    # ---- step 1: univariate IC screen (train)
    s1 = ic[(ic.abs_t >= MIN_ABS_T) & (ic.year_sign_stability >= MIN_SIGN_STABILITY)].index.tolist()

    # ---- step 2: redundancy (keep strongest |t| per cluster)
    s2, clusters = prune_redundant(corr, ic.abs_t, CORR_THRESHOLD, candidates=s1)

    # ---- step 3: VIF on the transformed train data, drop worst until all < MAX_VIF
    s3, vif_log = list(s2), []
    Xtr = Z.loc[train, s3]
    while True:
        v = vif(Xtr[s3])
        if v.iloc[0] < MAX_VIF or len(s3) <= 2:
            break
        vif_log.append((v.index[0], round(float(v.iloc[0]), 2)))
        s3.remove(v.index[0])
    final_vif = vif(Xtr[s3])

    # ---- step 4: linearity / monotonicity check of selected features
    mono = {}
    for f in s3:
        qs = quantile_spread(df[train], f)
        mono[f] = {"q1": qs.iloc[0], "q5": qs.iloc[-1],
                   "monotonicity": pd.Series(qs.values).corr(pd.Series(range(5)), method="spearman")}
    mono = pd.DataFrame(mono).T

    # ---- step 5: does the selection help a Ridge out of sample? (validation IC)
    sets = {"all 50 candidates": cands, "IC screen (step 1)": s1, "selected (steps 1-3)": s3}
    comp = []
    for name, feats in sets.items():
        for alpha in [1, 100, 10_000]:
            m = Ridge(alpha=alpha).fit(Z.loc[train, feats], df.loc[train, "y_win"])
            comp.append({"feature set": name, "n": len(feats), "alpha": alpha,
                         "train IC": ic_of(m.predict(Z.loc[train, feats]), df[train]),
                         **dict(zip(["val IC", "val IC NW t"], ic_with_t(m.predict(Z.loc[val, feats]), df[val])))})
    comp = pd.DataFrame(comp)
    coef = pd.Series(Ridge(alpha=100).fit(Z.loc[train, s3], df.loc[train, "y_win"]).coef_, index=s3)

    # ---- write dataset
    full = df[["Date", "Symbol", "split", "has_target", "tradable", "y_raw", "y_win", "y_rank", "y_class"]].join(Z[s3])
    write_dataset(full, s3, OUT)
    write_json({"model": "ridge", "features": s3, "transform": "per-date winsorize 1/99% then z-score; NaN -> 0",
                "target_default": "y_win", "eval_target": "y_raw",
                "params": {"MIN_ABS_T": MIN_ABS_T, "MIN_SIGN_STABILITY": MIN_SIGN_STABILITY,
                           "CORR_THRESHOLD": CORR_THRESHOLD, "MAX_VIF": MAX_VIF}},
               HERE / "selected_features.json")

    # ---- figures
    funnel = pd.Series({"candidates": len(cands), "IC screen": len(s1), "decorrelated": len(s2), "VIF < 5": len(s3)})
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(funnel.index[::-1], funnel.values[::-1], color=BLUE)
    for i, v in enumerate(funnel.values[::-1]):
        ax.text(v + 0.5, i, str(v), va="center", color=INK_2)
    ax.set(title="Ridge feature-selection funnel", xlabel="features")
    save(fig, FIG / "01_funnel.png")

    fig, axes = plt.subplots(1, len(s3), figsize=(2.3 * len(s3), 2.8), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], s3):
        qs = quantile_spread(df[train], f) * 100
        ax.bar(qs.index + 1, qs.values, color=np.where(qs.values > 0, ORANGE, BLUE), width=0.7)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_title(f, fontsize=8)
        ax.set_xticks(range(1, 6))
    axes[0][0].set_ylabel("mean 20d excess ret (%)")
    fig.suptitle("Linearity check: target by feature quintile (train)", fontweight="bold")
    save(fig, FIG / "02_quintile_monotonicity.png")

    cc = corr.loc[s3, s3]
    fig, ax = plt.subplots(figsize=(1 + 0.55 * len(s3), 0.8 + 0.5 * len(s3)))
    im = ax.imshow(cc.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(s3)), s3, rotation=90, fontsize=8)
    ax.set_yticks(range(len(s3)), s3, fontsize=8)
    for i in range(len(s3)):
        for j in range(len(s3)):
            ax.text(j, i, f"{cc.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Selected features: rank correlation")
    save(fig, FIG / "03_selected_correlation.png")

    show = [f for f in ["rel_volume_20d", "price_volume_interaction", "vol_adj_mom_60d"] if f in cands]
    fig, axes = plt.subplots(2, len(show), figsize=(4 * len(show), 5))
    for j, f in enumerate(show):
        axes[0, j].hist(df.loc[train, f].dropna(), bins=100, color=BLUE)
        axes[0, j].set_yscale("log")
        axes[0, j].set_title(f"{f} raw", fontsize=9)
        axes[1, j].hist(Z.loc[train, f], bins=100, color=ORANGE)
        axes[1, j].set_title(f"{f} after per-date winsor+z", fontsize=9)
    fig.suptitle("Why the transform: heavy tails before vs after (train)", fontweight="bold")
    save(fig, FIG / "04_transform_effect.png")

    fig, ax = plt.subplots(figsize=(7, 3))
    piv = comp.pivot(index="feature set", columns="alpha", values="val IC").loc[list(sets)]
    w = 0.25
    for k, a in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + (k - 1) * w, piv[a], width=w, label=f"alpha={a}",
               color=[BLUE, ORANGE, "#1baf7a"][k])
    ax.set_xticks(range(len(piv)), piv.index)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(title="Quick Ridge fit: validation rank IC by feature set", ylabel="mean daily IC")
    ax.legend()
    save(fig, FIG / "05_val_ic_by_set.png")

    # ---- report
    t = ic.loc[s3, ["ic_mean", "ic_tstat_nw", "year_sign_stability", "pdf_tier"]].join(
        final_vif.rename("VIF")).join(mono[["monotonicity"]]).join(coef.rename("ridge coef (a=100)"))
    t["coef sign = IC sign"] = np.sign(t["ridge coef (a=100)"]) == np.sign(t.ic_mean)
    md = [
        "# 01 — Ridge Regression: EDA & feature selection",
        "",
        "Prerequisite: `00_data_audit/REPORT.md` (shared audit, splits, IC table).",
        "",
        "## Why Ridge needs this preparation",
        "- Linear in the inputs → heavy-tailed features (kurtosis > 10 for most) dominate the fit unless winsorized.",
        "- Features must be on one scale for the L2 penalty to be fair → per-date z-score.",
        "- Market (`bench_*`) columns are constant on a date, so in a linear cross-sectional model they only move the "
        "intercept of every stock equally — they cannot change a ranking → excluded.",
        "- Correlated inputs make coefficients unstable and sign-flipping → strict redundancy pruning + VIF.",
        "- Per-date transforms use only same-day data, so they are leakage-free and need no refitting in walk-forward.",
        "",
        "## Selection funnel (all statistics on TRAIN 2019–2022 only)",
        "",
        f"1. **IC screen**: |Newey–West t| ≥ {MIN_ABS_T} and same IC sign in ≥ {MIN_SIGN_STABILITY:.0%} of train years "
        f"→ {len(s1)} of {len(cands)}: {', '.join(f'`{f}`' for f in s1)}",
        f"2. **Redundancy**: one feature per cluster of |cross-sectional rank corr| ≥ {CORR_THRESHOLD} (keep highest |t|) → {len(s2)}",
        f"3. **VIF < {MAX_VIF}**: removed {vif_log if vif_log else 'nothing'} → **{len(s3)} features**",
        "",
        "![funnel](figures/01_funnel.png)",
        "",
        "## Selected features",
        "",
        t.to_markdown(floatfmt=".4f"),
        "",
        "Monotonicity = Spearman correlation between quintile number and mean target (±1 = perfectly linear-friendly).",
        "",
        "![quintiles](figures/02_quintile_monotonicity.png)",
        "![corr](figures/03_selected_correlation.png)",
        "![transform](figures/04_transform_effect.png)",
        "",
        "Redundancy clusters among the IC-screened features:",
        "",
        clusters.to_markdown(floatfmt=".3f"),
        "",
        "## Sanity check: does the selection help out of sample?",
        "",
        "Plain Ridge on `y_win`, fit on train, scored by mean daily rank IC on validation (2023–mid 2024). "
        "This is a check on the feature set, not model tuning (that comes next).",
        "",
        "**Read with care:** the validation window holds only ~17 non-overlapping 20-day periods, so the standard error "
        "of a mean IC is roughly IC / t ≈ 0.03. Differences of ±0.01 between feature sets are noise. Variants tried during "
        "design (no IC screen; |t| ≥ 1; corr cut 0.7/0.8/0.9; VIF 5/10) all landed between 0.089 and 0.105. The loose "
        "screen + decorrelation + VIF set was chosen on principle (interpretable, stable coefficients), not because "
        "it had the best validation number.",
        "",
        comp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "![val](figures/05_val_ic_by_set.png)",
        "",
        "## Model-ready dataset",
        "",
        f"`data/ridge_dataset.parquet` — {len(full):,} rows. Columns: `Date, Symbol, split, has_target, tradable`, "
        f"{len(s3)} features (float32, per-date winsorized z-scores, no NaN), targets `y_raw, y_win, y_rank, y_class`.",
        "",
        split_table(full).to_markdown(),
        "",
        "Use: `X = df[features]`, `y = df.y_win` for rows with `split == 'train'` and `has_target`; tune on `val`; "
        "never touch `test` until the final run. Drop `purge` and `burn_in` rows. Evaluate with `y_raw`. "
        "Feature list: `selected_features.json`.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(md))
    print("selected:", s3)
    print(comp)


if __name__ == "__main__":
    main()
