"""ElasticNet — EDA, feature selection and model-ready dataset.

    .venv/bin/python 02_elasticnet/prepare_data.py

ElasticNet does embedded selection (L1) while the L2 part keeps groups of
correlated features together. So the pre-screen is deliberately light: remove
only near-duplicates, let the penalty choose the rest, and measure how *stable*
that choice is with block-bootstrap stability selection.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sklearn.linear_model import ElasticNet, enet_path  # noqa: E402

from common import config as C  # noqa: E402
from common.data import cs_winsor_zscore, load_clean  # noqa: E402
from common.plotting import BLUE, INK_2, ORANGE, SERIES, plt, save  # noqa: E402
from common.prep import add_targets, every_kth_date, ic_with_t, split_table, write_dataset, write_json  # noqa: E402
from common.selection import prune_redundant  # noqa: E402

FIG = HERE / "figures"
OUT = HERE / "data" / "elasticnet_dataset.parquet"

CORR_THRESHOLD = 0.90     # remove near-duplicates only
L1_RATIOS = [0.1, 0.5, 0.9]
ALPHAS = np.logspace(-5, -2, 13)
N_BOOT, BLOCK_DAYS, STABLE_FREQ, MAX_Q = 40, 60, 0.6, 15


def year_folds(frame):
    """Leave-one-train-year-out folds with a 20-day purge on each side."""
    years = frame.Date.dt.year
    cal = np.sort(frame.Date.unique())
    for y in sorted(years.unique()):
        te = years == y
        lo, hi = frame.Date[te].min(), frame.Date[te].max()
        i_lo, i_hi = np.searchsorted(cal, lo), np.searchsorted(cal, hi)
        gap_lo = cal[max(i_lo - C.HORIZON, 0)]
        gap_hi = cal[min(i_hi + C.HORIZON, len(cal) - 1)]
        tr = (frame.Date < gap_lo) | (frame.Date > gap_hi)
        yield y, tr.to_numpy(), te.to_numpy()


def main():
    df = add_targets(load_clean())
    ic = pd.read_csv(C.FEATURE_IC, index_col=0)
    corr = pd.read_csv(C.FEATURE_CORR, index_col=0)
    cands = list(C.STOCK_CANDIDATES)

    Z = cs_winsor_zscore(df, cands).fillna(0.0)
    train = df.split.eq("train") & df.has_target
    val = df.split.eq("val") & df.has_target

    # ---- step 1: near-duplicate removal only
    s1, clusters = prune_redundant(corr, ic.abs_t, CORR_THRESHOLD, candidates=cands)
    dropped = [f for f in cands if f not in s1]

    # thin to every 5th date for the CV / bootstrap fits (targets overlap 19/20 days anyway)
    thin = train & every_kth_date(df[train], 5).reindex(df.index, fill_value=False)
    Xt, yt, ft = Z.loc[thin, s1], df.loc[thin, "y_win"], df.loc[thin, ["Date"]]

    # ---- step 2: purged leave-one-year-out CV over (alpha, l1_ratio), scored by rank IC
    cv = []
    for l1 in L1_RATIOS:
        for a in ALPHAS:
            ics, nnz = [], []
            for y, tr, te in year_folds(ft):
                m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000).fit(Xt[tr], yt[tr])
                p = m.predict(Xt[te])
                ics.append(0.0 if np.allclose(p, p[0]) else
                           ic_with_t(p, df.loc[thin][te])[0])
                nnz.append((m.coef_ != 0).sum())
            cv.append({"l1_ratio": l1, "alpha": a, "cv_ic": np.mean(ics), "n_nonzero": np.mean(nnz)})
    cv = pd.DataFrame(cv)
    best = cv.loc[cv.cv_ic.idxmax()]
    alpha, l1 = float(best.alpha), float(best.l1_ratio)

    # ---- regularisation path on the thinned train set (l1_ratio = best)
    alphas_p, coefs, _ = enet_path(Xt.to_numpy(), yt.to_numpy(), l1_ratio=l1,
                                   alphas=np.logspace(-5, -1.5, 40))
    entry = {f: (alphas_p[np.nonzero(coefs[i])[0]].max() if coefs[i].any() else 0.0)
             for i, f in enumerate(s1)}
    n_active = (coefs != 0).sum(axis=0)

    # ---- step 3: stability selection (Meinshausen & Buhlmann 2010)
    # Penalty region: from the best-CV alpha down to where MAX_Q features are active.
    region = alphas_p[(alphas_p <= alpha) & (n_active <= MAX_Q)]
    if len(region) == 0:
        region = np.array([alpha])
    rng = np.random.default_rng(C.SEED)
    cal = np.sort(ft.Date.unique())
    blocks = np.arange(len(cal)) // (BLOCK_DAYS // 5)
    ub = np.unique(blocks)
    hits = np.zeros((len(s1), len(region)))
    union_sizes = []
    for b in range(N_BOOT):
        chosen = rng.choice(ub, size=len(ub) // 2, replace=False)
        mask = ft.Date.isin(cal[np.isin(blocks, chosen)]).to_numpy()
        _, cb, _ = enet_path(Xt[mask].to_numpy(), yt[mask].to_numpy(), l1_ratio=l1, alphas=region)
        nz = cb != 0
        hits += nz
        union_sizes.append(nz.any(axis=1).sum())
    freq = pd.Series((hits / N_BOOT).max(axis=1), index=s1).sort_values(ascending=False)
    stable = freq[freq >= STABLE_FREQ].index.tolist()
    q = float(np.mean(union_sizes))
    ev_bound = q ** 2 / ((2 * STABLE_FREQ - 1) * len(s1))

    # ---- validation check
    comp = []
    for name, feats in {"all 50 candidates": cands, f"pre-screened (corr<{CORR_THRESHOLD})": s1,
                        f"stable (freq>={STABLE_FREQ})": stable}.items():
        m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000).fit(Z.loc[train, feats], df.loc[train, "y_win"])
        v, t = ic_with_t(m.predict(Z.loc[val, feats]), df[val])
        comp.append({"feature set": name, "n in": len(feats), "n nonzero": int((m.coef_ != 0).sum()),
                     "val IC": v, "val IC NW t": t})
    comp = pd.DataFrame(comp)
    final = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000).fit(Z.loc[train, s1], df.loc[train, "y_win"])
    coef = pd.Series(final.coef_, index=s1)

    # ---- write dataset: the pre-screened set (EN does the final selection itself)
    full = df[["Date", "Symbol", "split", "has_target", "tradable", "y_raw", "y_win", "y_rank", "y_class"]].join(Z[s1])
    write_dataset(full, s1, OUT)
    write_json({"model": "elasticnet", "features": s1, "stable_features": stable,
                "transform": "per-date winsorize 1/99% then z-score; NaN -> 0",
                "target_default": "y_win", "eval_target": "y_raw",
                "cv_suggested": {"alpha": alpha, "l1_ratio": l1},
                "params": {"CORR_THRESHOLD": CORR_THRESHOLD, "N_BOOT": N_BOOT, "MAX_Q": MAX_Q,
                           "BLOCK_DAYS": BLOCK_DAYS, "STABLE_FREQ": STABLE_FREQ}},
               HERE / "selected_features.json")

    # ---- figures
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k, l in enumerate(L1_RATIOS):
        c = cv[cv.l1_ratio == l]
        ax.plot(c.alpha, c.cv_ic, marker="o", ms=4, color=SERIES[k], label=f"l1_ratio={l}")
    ax.axvline(alpha, color=INK_2, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set(title="Purged leave-one-year-out CV (train)", xlabel="alpha", ylabel="mean rank IC")
    ax.legend()
    save(fig, FIG / "01_cv_curve.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    order = sorted(s1, key=lambda f: -entry[f])
    for i, f in enumerate(order[:8]):
        ax.plot(alphas_p, coefs[s1.index(f)], color=SERIES[i], lw=1.6, label=f)
    for f in order[8:]:
        ax.plot(alphas_p, coefs[s1.index(f)], color="#c3c2b7", lw=0.8)
    ax.axvline(alpha, color=INK_2, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(title=f"ElasticNet path (l1_ratio={l1}); first 8 to enter coloured",
           xlabel="alpha (strong → weak penalty)", ylabel="coefficient")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    save(fig, FIG / "02_regularisation_path.png")

    fig, ax = plt.subplots(figsize=(7, 9))
    f2 = freq.sort_values()
    ax.barh(f2.index, f2.values, color=np.where(f2.values >= STABLE_FREQ, ORANGE, BLUE))
    ax.axvline(STABLE_FREQ, color=INK_2, ls="--", lw=0.8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set(title=f"Stability selection: {N_BOOT} block bootstraps", xlabel="max selection probability over penalty region")
    save(fig, FIG / "03_stability.png")

    # ---- report
    t = pd.DataFrame({"selection_freq": freq, "entry_alpha": pd.Series(entry), "coef (train fit)": coef}).join(
        ic[["ic_mean", "ic_tstat_nw", "pdf_tier"]]).sort_values("selection_freq", ascending=False)
    md = [
        "# 02 — ElasticNet: EDA & feature selection",
        "",
        "Prerequisite: `00_data_audit/REPORT.md`. Same transform as Ridge (per-date winsorize + z-score, market columns excluded).",
        "",
        "## Why ElasticNet is prepared differently from Ridge",
        "- The L1 term selects features itself, so a hard filter would pre-empt what the model is for. We remove only "
        f"near-duplicates (|rank corr| ≥ {CORR_THRESHOLD}) — e.g. `mom_k` vs `rel_mom_k` are the same feature within a date.",
        "- Lasso picks one member of a correlated group arbitrarily; ElasticNet's L2 part spreads weight across the group. "
        "The stability analysis shows which picks are reliable.",
        "",
        f"## Step 1 — near-duplicates removed ({len(dropped)}): " + ", ".join(f"`{f}`" for f in dropped),
        "",
        f"→ **{len(s1)} pre-screened features** go into the dataset.",
        "",
        "## Step 2 — penalty strength (purged leave-one-year-out CV on train, every 5th date)",
        "",
        f"Best CV IC {best.cv_ic:.4f} at **alpha={alpha:.2g}, l1_ratio={l1}** ({best.n_nonzero:.0f} non-zero on average). "
        "The curve is flat: models with 5 and with 39 active features score within ~0.002 IC of each other, i.e. the data "
        "cannot distinguish them. The suggested alpha is a starting point for the modelling step, not a final tuning.",
        "",
        "![cv](figures/01_cv_curve.png)",
        "![path](figures/02_regularisation_path.png)",
        "",
        f"## Step 3 — stability selection ({N_BOOT} bootstraps of half the {BLOCK_DAYS}-day blocks)",
        "",
        f"Meinshausen–Bühlmann: selection probability = max over the penalty region alpha ∈ [{region.min():.2g}, {region.max():.2g}] "
        f"(best-CV alpha down to ≤ {MAX_Q} active features). Average union size q = {q:.1f}, so the expected number of "
        f"falsely selected features at threshold {STABLE_FREQ} is ≤ q²/((2π−1)p) = **{ev_bound:.2f}**.",
        "",
        f"**{len(stable)} features selected in ≥ {STABLE_FREQ:.0%} of resamples**: " + ", ".join(f"`{f}`" for f in stable),
        "",
        "![stability](figures/03_stability.png)",
        "",
        t.to_markdown(floatfmt=".4g"),
        "",
        "## Sanity check on validation (fit on train at the suggested alpha / l1_ratio)",
        "",
        comp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Validation holds ~17 independent 20-day periods: an IC standard error of ~0.03 means small differences are noise.",
        "",
        "## Model-ready dataset",
        "",
        f"`data/elasticnet_dataset.parquet` — {len(full):,} rows, {len(s1)} pre-screened features (float32, no NaN), "
        "targets `y_raw, y_win, y_rank, y_class`.",
        "",
        split_table(full).to_markdown(),
        "",
        "Recommended use: feed all `features` from `selected_features.json` and let the penalty select; "
        "`stable_features` is the conservative alternative for an ablation. Train on `split == 'train'`, tune on `val`, "
        "evaluate on `y_raw`.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(md))
    print(cv.sort_values("cv_ic", ascending=False).head(5))
    print("pick", alpha, l1, "stable", stable)
    print(comp)


if __name__ == "__main__":
    main()
