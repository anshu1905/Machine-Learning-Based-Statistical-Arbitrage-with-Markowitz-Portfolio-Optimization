"""Shared preparation for the sequence models (RNN, LSTM).

A sample is (stock s, date t): input = the stock's feature vectors on the
`lookback` trading days ending at t, label = 20d excess return from t.
Data are stored once as dense panels [stock, date, feature] plus a sample
index, so windows are sliced on the fly instead of materialising
N x lookback x F copies (which would be several GB).
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import config as C
from .data import cs_gauss_rank
from .prep import every_kth_date, ic_with_t
from .plotting import BLUE, INK_2, ORANGE, SERIES, plt, save
from .prep import add_targets, split_table
from .data import load_clean
from .selection import daily_rank_ic, prune_redundant

MARKET_CLIP = 5.0


def transform(df, stock_feats, market_feats):
    """Stock: per-date Gaussian rank (bounded, ~N(0,1), leakage-free), NaN -> 0.
    Market: z-score with TRAIN mean/std, clipped to +-5, NaN -> 0."""
    Xs = cs_gauss_rank(df, stock_feats).fillna(0.0)
    daily = df.loc[df.split.eq("train")].drop_duplicates("Date")[market_feats]
    mu, sd = daily.mean(), daily.std()
    Xm = ((df[market_feats] - mu) / sd).clip(-MARKET_CLIP, MARKET_CLIP).fillna(0.0)
    scaler = {f: {"mean": float(mu[f]), "std": float(sd[f])} for f in market_feats}
    return pd.concat([Xs, Xm], axis=1), scaler


def to_panel(df, X, cols):
    symbols = np.array(sorted(df.Symbol.unique()))
    dates = np.sort(df.Date.unique())
    s = np.searchsorted(symbols, df.Symbol.to_numpy())
    t = np.searchsorted(dates, df.Date.to_numpy())
    P = np.zeros((len(symbols), len(dates), len(cols)), dtype=np.float32)
    P[s, t] = X[cols].to_numpy(np.float32)
    present = np.zeros((len(symbols), len(dates)), dtype=bool)
    present[s, t] = True
    return P, present, symbols, dates, s, t


def persistence(P, present, date_mask, lags=(1, 5, 20, 60)):
    """Pooled autocorrelation of each feature at several lags (train dates only)."""
    out = {}
    for k in lags:
        ok = present[:, k:] & present[:, :-k] & date_mask[k:][None, :]
        a, b = P[:, k:][ok], P[:, :-k][ok]
        a = (a - a.mean(0)) / (a.std(0) + 1e-12)
        b = (b - b.mean(0)) / (b.std(0) + 1e-12)
        out[f"lag_{k}"] = (a * b).mean(0)
    return pd.DataFrame(out)


def lagged_ic(df, feats, lags):
    """IC of feature value k days BEFORE t against the target from t (train)."""
    tr = df[df.split.eq("train") & df.has_target]
    out = {}
    g = df.groupby("Symbol")[feats]
    for k in lags:
        shifted = (g.shift(k) if k else df[feats]).loc[tr.index]
        tmp = pd.concat([tr[["Date", C.TARGET]], shifted], axis=1)
        out[k] = daily_rank_ic(tmp, feats).mean()
    return pd.DataFrame(out)


def history_probe(df, X, feats, lags, alpha=100.0):
    """Linear probe: does adding past values of the features raise validation IC?
    Ridge on [x_t, x_{t-k} for k in lags] vs [x_t] alone."""
    g = pd.concat([df[["Symbol"]], X[feats]], axis=1).groupby("Symbol")[feats]
    train = df.split.eq("train") & df.has_target
    train = train & every_kth_date(df[train], 5).reindex(df.index, fill_value=False)
    val = df.split.eq("val") & df.has_target
    rows, cols = [], [X[feats]]
    for k in [0] + list(lags):
        if k:
            cols.append(g.shift(k).fillna(0.0).add_suffix(f"_l{k}"))
        D = pd.concat(cols, axis=1)
        m = Ridge(alpha=alpha).fit(D[train], df.loc[train, "y_win"])
        v, t = ic_with_t(m.predict(D[val]), df[val])
        rows.append({"history up to lag": k, "inputs": D.shape[1], "val IC": v, "val IC NW t": t})
    return pd.DataFrame(rows)


def sample_index(df, s, t, present, lookback):
    """One row per (stock, date) usable as a sample: date after the burn-in and the
    stock was listed for the whole window."""
    cum = np.cumsum(present, axis=1)
    window_full = np.zeros_like(present)
    window_full[:, lookback - 1:] = (cum[:, lookback - 1:] -
                                     np.concatenate([np.zeros((present.shape[0], 1), int),
                                                     cum[:, :-lookback]], axis=1)) == lookback
    ok = window_full[s, t] & df.split.isin(["train", "purge", "val", "test"]).to_numpy()
    idx = pd.DataFrame({"s_idx": s, "t_idx": t, "Date": df.Date.to_numpy(), "Symbol": df.Symbol.to_numpy(),
                        "split": df.split.to_numpy(), "has_target": df.has_target.to_numpy(),
                        "tradable": df.tradable.to_numpy()})[ok]
    return idx.reset_index(drop=True)


def save_arrays(out_dir, P, df, s, t, present, symbols, dates, feats, samples, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_panel.npy", P)
    Y = {}
    for col in ["y_raw", "y_win", "y_rank", "y_class"]:
        A = np.full(present.shape, np.nan, dtype=np.float32)
        A[s, t] = df[col].to_numpy(np.float32)
        Y[col] = A
    np.savez(out_dir / "targets.npz", **Y, present=present)
    samples.to_parquet(out_dir / "samples.parquet", index=False)
    meta = {**meta, "features": feats, "symbols": symbols.tolist(),
            "dates": [str(pd.Timestamp(d).date()) for d in dates],
            "X_panel_shape": list(P.shape)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def run(here, name, lookback, n_core, corr_threshold, max_fast, market, probe_lags, rationale):
    """Full EDA + selection + export for one sequence model folder."""
    fig_dir, out_dir = here / "figures", here / "data"
    df = add_targets(load_clean())
    ic = pd.read_csv(C.FEATURE_IC, index_col=0)
    corr = pd.read_csv(C.FEATURE_CORR, index_col=0)

    # ---- persistence of every candidate (train dates)
    Xall, _ = transform(df, C.STOCK_CANDIDATES, market)
    P_all, present, symbols, dates, s, t = to_panel(df, Xall, C.STOCK_CANDIDATES)
    train_dates = np.isin(dates, df.loc[df.split.eq("train"), "Date"].unique())
    pers = persistence(P_all, present, train_dates).set_axis(C.STOCK_CANDIDATES)
    del P_all

    # ---- selection: core (IC after redundancy) + fast daily features + market regime
    pruned, clusters = prune_redundant(corr, ic.abs_t, corr_threshold, candidates=C.STOCK_CANDIDATES)
    core = ic.loc[pruned].sort_values("abs_t", ascending=False).index[:n_core].tolist()
    fast_pool = pers.loc[pruned].query("lag_1 < 0.5").sort_values("lag_1").index.tolist()
    fast = [f for f in fast_pool if f not in core][:max_fast]
    stock = core + fast
    feats = stock + market

    # ---- information decay and history probe
    lags = [0, 1, 5, 10, 20, 40, 60]
    lic = lagged_ic(df, stock, lags)
    X, scaler = transform(df, stock, market)
    probe = history_probe(df, X, stock, probe_lags)

    # ---- export
    P, present, symbols, dates, s, t = to_panel(df, X, feats)
    samples = sample_index(df, s, t, present, lookback)
    save_arrays(out_dir, P, df, s, t, present, symbols, dates, feats, samples, {
        "model": name, "lookback": lookback, "stock_features": stock, "core_features": core,
        "fast_features": fast, "market_features": market, "market_scaler_train": scaler,
        "transform": "stock: per-date Gaussian rank, NaN -> 0; market: train z-score clipped +-5, NaN -> 0",
        "target_default": "y_win", "eval_target": "y_raw", "horizon": C.HORIZON})
    (here / "selected_features.json").write_text(json.dumps(
        {"model": name, "features": feats, "lookback": lookback, "core": core, "fast": fast,
         "market": market}, indent=2))

    # ---- figures
    pp = pers.loc[stock].sort_values("lag_1")
    fig, ax = plt.subplots(figsize=(8, 0.8 + 0.32 * len(pp)))
    y = np.arange(len(pp))
    for k, c in enumerate(pp.columns):
        ax.scatter(pp[c], y, color=SERIES[k], s=28, label=c.replace("_", " "), zorder=3)
    ax.set_yticks(y, pp.index, fontsize=8)
    ax.axvline(0.5, color=INK_2, ls="--", lw=0.8)
    ax.set(title="Feature persistence: autocorrelation of per-date rank (train)", xlabel="autocorrelation")
    ax.legend(ncol=4, loc="lower left", fontsize=8)
    save(fig, fig_dir / "01_persistence.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    top = lic.abs().mean(axis=1).sort_values(ascending=False).index[:8]
    for k, f in enumerate(top):
        ax.plot(lags, lic.loc[f], marker="o", ms=4, color=SERIES[k], label=f)
    ax.plot(lags, lic.abs().mean(), color=INK_2, lw=2.5, ls="--", label="mean |IC| (all selected)")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.axvline(lookback, color=ORANGE, lw=1, ls=":")
    ax.set(title=f"Information decay: IC of feature k days before t (lookback = {lookback})",
           xlabel="k (trading days)", ylabel="mean daily IC")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, fig_dir / "02_lagged_ic.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(probe["history up to lag"], probe["val IC"], marker="o", color=BLUE)
    ax.set(title="Linear probe: validation IC with feature history", xlabel="history included up to lag (days)",
           ylabel="val IC")
    save(fig, fig_dir / "03_history_probe.png")

    n_by = samples.groupby("split").agg(samples=("s_idx", "size"), with_target=("has_target", "sum"))
    md = [
        f"# {name} — EDA & feature selection",
        "",
        "Prerequisite: `00_data_audit/REPORT.md`.",
        "",
        *rationale,
        "",
        "## Selection",
        "",
        f"1. Redundancy pruning at |cross-sectional rank corr| ≥ {corr_threshold} → {len(pruned)} stock features.",
        f"2. **Core** = the {n_core} strongest by |Newey–West IC t| (train): " + ", ".join(f"`{f}`" for f in core),
        f"3. **Fast** = up to {max_fast} daily-varying features (lag-1 autocorrelation < 0.5) not already in core. "
        "Their univariate IC is small, but their *path* over the window (reversal, volume shocks, gaps) is exactly what a "
        "recurrent model can use and a cross-sectional model cannot: " + (", ".join(f"`{f}`" for f in fast) or "none"),
        "4. **Market regime** inputs (same for all stocks on a date, z-scored with train statistics): "
        + ", ".join(f"`{f}`" for f in market),
        "",
        f"→ **{len(feats)} input channels, lookback {lookback} trading days.**",
        "",
        "## Persistence",
        "",
        "Slow features (autocorrelation ≈ 1 even at lag 20) repeat almost the same value at every step of a window; "
        "they mainly tell the network the current state. Fast features change daily and carry the temporal pattern.",
        "",
        "![persistence](figures/01_persistence.png)",
        "",
        pers.loc[stock].to_markdown(floatfmt=".3f"),
        "",
        "## How far back is there information? (lookback choice)",
        "",
        "![lagged](figures/02_lagged_ic.png)",
        "",
        lic.to_markdown(floatfmt=".4f"),
        "",
        "## Does feature history add information beyond today's values?",
        "",
        "Linear probe (Ridge on today's features plus lagged copies, train → validation). A sequence model should beat "
        "the lag-0 row; if history adds nothing linearly, any RNN/LSTM gain must come from non-linear temporal patterns.",
        "",
        probe.to_markdown(index=False, floatfmt=".4f"),
        "",
        "![probe](figures/03_history_probe.png)",
        "",
        "## Model-ready dataset (`data/`)",
        "",
        f"- `X_panel.npy` float32 **{list(P.shape)}** = [stock, trading day, feature]; zeros where a stock was not listed.",
        "- `targets.npz`: `y_raw, y_win, y_rank, y_class` as [stock, day] (NaN where unavailable) and `present` mask.",
        f"- `samples.parquet`: one row per usable (stock, day) — window of {lookback} listed days fully inside the data; "
        "columns `s_idx, t_idx, Date, Symbol, split, has_target, tradable`.",
        "- `meta.json`: feature order, symbols, dates, lookback, train-fitted market scaler.",
        "",
        n_by.to_markdown(),
        "",
        "Load with the ready-made PyTorch dataset:",
        "",
        "```python",
        "from dataset import load",
        "train_ds, val_ds, test_ds = load()           # SequenceDataset objects",
        "x, y = train_ds[0]                           # x: [lookback, n_features], y: scalar",
        "```",
        "",
        "Windows look back into earlier periods (e.g. a January-2023 validation window uses December-2022 features). "
        "That is legitimate — those are past inputs — and the 20-day purge prevents any *target* overlap.",
    ]
    (here / "REPORT.md").write_text("\n".join(md))
    print(name, "features", feats)
    print(probe)
    print(n_by)
