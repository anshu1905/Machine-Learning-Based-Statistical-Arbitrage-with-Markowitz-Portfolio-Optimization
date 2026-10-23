"""Training engine for the sequence models (RNN, LSTM).

Speed: the whole [stock, day, feature] panel (~45 MB) lives on the GPU and a
batch of windows is gathered with one indexing op, X[s, t-L+1 .. t], so there
is no Python DataLoader in the training loop.

Batches are groups of whole dates (all stocks of `dates_per_batch` days). That
matches how the model is used (rank stocks within a date) and allows the
cross-sectional IC loss.

Protocol is the same as the tabular models (see common/training.py):
tune on train -> validation IC, early stopping on validation IC, then
walk-forward refits over test with the tuned config and the tuned number of
epochs, averaging several seeds.
"""
import argparse
import contextlib
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config as C
from .evaluate import finish
from .prep import ic_with_t
from .training import expand_grid, safe_score, train_mask_for, wf_blocks

TARGETS = ["y_raw", "y_win", "y_rank", "y_class"]
LOSS_TARGET = {"mse": "y_win", "ic": "y_rank"}


# ------------------------------------------------------------------ device / io
def pick_device(arg="auto"):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Panel:
    """X_panel.npy + targets.npz + samples.parquet of one model folder, on `device`."""

    def __init__(self, data_dir, device):
        self.meta = json.loads((data_dir / "meta.json").read_text())
        self.lookback = self.meta["lookback"]
        self.device = device
        self.X = torch.from_numpy(np.load(data_dir / "X_panel.npy")).to(device)
        T = np.load(data_dir / "targets.npz")
        s = pd.read_parquet(data_dir / "samples.parquet")
        for k in TARGETS:
            s[k] = T[k][s.s_idx.to_numpy(), s.t_idx.to_numpy()]
        self.samples = s
        self.offsets = torch.arange(-self.lookback + 1, 1, device=device)
        self.n_features = self.X.shape[-1]

    def windows(self, s, t):
        return self.X[s[:, None], t[:, None] + self.offsets[None, :]]   # [B, L, F]


# ------------------------------------------------------------------------ model
class SeqNet(nn.Module):
    """Recurrent encoder over the window -> last hidden state -> linear score."""

    def __init__(self, n_features, cell="lstm", hidden=64, layers=1, dropout=0.1):
        super().__init__()
        rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


def ic_loss(pred, y, groups):
    """-mean over dates of Pearson corr(pred, y) inside each date."""
    n = torch.bincount(groups).clamp(min=1).float()
    mean = lambda v: torch.zeros_like(n).index_add_(0, groups, v) / n  # noqa: E731
    pc, yc = pred - mean(pred)[groups], y - mean(y)[groups]
    cov = torch.zeros_like(n).index_add_(0, groups, pc * yc)
    vp = torch.zeros_like(n).index_add_(0, groups, pc * pc)
    vy = torch.zeros_like(n).index_add_(0, groups, yc * yc)
    return -(cov / (vp.sqrt() * vy.sqrt() + 1e-8)).mean()


# --------------------------------------------------------------------- training
class Rows:
    """A set of samples as device tensors."""

    def __init__(self, panel, frame, target):
        dev = panel.device
        self.frame = frame
        self.s = torch.as_tensor(frame.s_idx.to_numpy().copy(), device=dev)
        self.t = torch.as_tensor(frame.t_idx.to_numpy().copy(), device=dev)
        self.y = torch.as_tensor(frame[target].to_numpy(np.float32), device=dev)
        codes, self.dates = pd.factorize(frame.t_idx.to_numpy())
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(len(self.dates) + 1))
        self.by_date = [order[bounds[i]:bounds[i + 1]] for i in range(len(self.dates))]
        self.date_code = torch.as_tensor(codes, device=dev)
        self.per_date = min(len(g) for g in self.by_date)

    def batches(self, dates_per_batch, rng):
        """Fixed-size batches: `dates_per_batch` random dates x `per_date` random stocks each.
        Constant shapes keep GPU kernels/caches stable and weight every date equally."""
        perm = rng.permutation(len(self.by_date))
        n = max(len(perm) // dates_per_batch, 1) * dates_per_batch
        for i in range(0, n, dates_per_batch):
            yield np.concatenate([rng.choice(self.by_date[j], self.per_date, replace=False)
                                  for j in perm[i:i + dates_per_batch]])


@torch.no_grad()
def predict(model, panel, rows, batch=16384):
    model.eval()
    out = []
    for i in range(0, len(rows.s), batch):
        s, t = rows.s[i:i + batch], rows.t[i:i + batch]
        with _autocast(panel.device):
            out.append(model(panel.windows(s, t)).float())
    return torch.cat(out).cpu().numpy()


def _autocast(device):
    """fp16 mixed precision on CUDA (cuDNN RNN/LSTM kernels support it); full precision elsewhere."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def fit(panel, cfg, train, val=None, epochs=40, patience=6, seed=0, verbose=True):
    """Train one network. With `val`: early stopping on validation rank IC.
    Returns (model with best weights, info)."""
    set_seed(seed)
    dev = panel.device
    model = SeqNet(panel.n_features, cfg["cell"], cfg["hidden"], cfg["layers"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    rng = np.random.default_rng(seed)
    best = {"ic": -np.inf, "epoch": epochs, "state": None}
    history, bad = [], 0

    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for idx in train.batches(cfg.get("dates_per_batch", 8), rng):
            idx = torch.as_tensor(idx, device=dev)
            x = panel.windows(train.s[idx], train.t[idx])
            with _autocast(dev):
                p = model(x).float()
            y = train.y[idx]
            if cfg["loss"] == "ic":
                loss = ic_loss(p, y, torch.unique(train.date_code[idx], return_inverse=True)[1])
            else:
                loss = nn.functional.mse_loss(p, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # recurrent nets explode without it
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if val is not None:
            ic, _ = ic_with_t(predict(model, panel, val), val.frame)
            rec["val_ic"] = ic
            if safe_score(ic) > best["ic"]:
                best.update(ic=ic, epoch=ep, state=copy.deepcopy(model.state_dict()))
                bad = 0
            else:
                bad += 1
        history.append(rec)
        if verbose:
            print(f"    epoch {ep:>2}  loss {rec['train_loss']:.5f}  val IC {rec.get('val_ic', np.nan):.4f}  "
                  f"{rec['seconds']}s", flush=True)
        if val is not None and bad >= patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "val_ic": best["ic"], "history": history}


# ------------------------------------------------------------------- full run
def base_args(description, cell):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu")
    p.add_argument("--mode", choices=["wf", "fixed"], default="wf")
    p.add_argument("--retrain-every", type=int, default=63)
    p.add_argument("--window-years", type=float, default=None)
    p.add_argument("--n-iter", type=int, default=12, help="random configs from the grid")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seeds", type=int, default=3, help="ensemble size for the final predictions")
    p.add_argument("--quick", action="store_true", help="smoke test")
    p.set_defaults(cell=cell)
    return p


def run(folder, name, grid, args):
    dev = pick_device(args.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"{name} on {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""))
    panel = Panel(folder / "data", dev)
    S = panel.samples
    if args.quick:
        args.epochs, args.patience, args.seeds, args.n_iter = 2, 2, 2, 2

    tr_frame = S[S.split.eq("train") & S.has_target]
    va_frame = S[S.split.eq("val") & S.has_target]
    if args.quick:   # smoke test on a slice of train
        tr_frame = tr_frame[tr_frame.t_idx >= tr_frame.t_idx.max() - 120]

    # ---- 1. tune
    combos = expand_grid(grid, args.n_iter)
    rows, best = [], None
    for k, cfg in enumerate(combos, 1):
        cfg = {"cell": args.cell, **cfg}
        target = LOSS_TARGET[cfg["loss"]]
        print(f"[{k}/{len(combos)}] {cfg}", flush=True)
        t0 = time.time()
        model, info = fit(panel, cfg, Rows(panel, tr_frame, target), Rows(panel, va_frame, target),
                          args.epochs, args.patience, seed=0)
        rows.append({**cfg, "val_ic": info["val_ic"], "best_epoch": info["best_epoch"],
                     "seconds": round(time.time() - t0, 1)})
        if best is None or safe_score(info["val_ic"]) > safe_score(best["val_ic"]):
            best = {"cfg": cfg, "val_ic": info["val_ic"], "epochs": [info["best_epoch"]],
                    "val_preds": [predict(model, panel, Rows(panel, va_frame, target))]}
    tuning = pd.DataFrame(rows).sort_values("val_ic", ascending=False)
    cfg, target = best["cfg"], LOSS_TARGET[best["cfg"]["loss"]]

    # ---- 2. extra seeds of the best config (validation ensemble + epoch count)
    for seed in range(1, args.seeds):
        print(f"best config, seed {seed}", flush=True)
        model, info = fit(panel, cfg, Rows(panel, tr_frame, target), Rows(panel, va_frame, target),
                          args.epochs, args.patience, seed=seed, verbose=False)
        best["epochs"].append(info["best_epoch"])
        best["val_preds"].append(predict(model, panel, Rows(panel, va_frame, target)))
    n_epochs = max(1, int(round(np.mean(best["epochs"]))))

    # ---- 3. walk-forward over test
    elig = S.has_target                      # samples start after the burn-in
    test = S.split.eq("test")
    blocks = wf_blocks(S.loc[test, "t_idx"], args.retrain_every, args.mode, max_blocks=2 if args.quick else None)
    test_pred = pd.Series(np.nan, index=S.index)
    wf_log = []
    for k, (i0, i1) in enumerate(blocks, 1):
        m = elig & train_mask_for(S.t_idx, i0, args.window_years)
        pm = test & (S.t_idx >= i0) & (S.t_idx < i1)
        t0 = time.time()
        preds = []
        for seed in range(args.seeds):
            model, _ = fit(panel, cfg, Rows(panel, S[m], target), None, n_epochs, seed=seed, verbose=False)
            preds.append(predict(model, panel, Rows(panel, S[pm], target)))
        test_pred[pm] = np.mean(preds, axis=0)
        wf_log.append({"refit_date": str(S.loc[pm, "Date"].min())[:10], "predict_to": str(S.loc[pm, "Date"].max())[:10],
                       "train_rows": int(m.sum()), "epochs": n_epochs, "seeds": args.seeds,
                       "seconds": round(time.time() - t0, 1)})
        print(f"walk-forward {k}/{len(blocks)}: {wf_log[-1]}", flush=True)

    cols = ["Date", "Symbol", "split", "tradable", "y_raw"]
    out = pd.concat([va_frame[cols].assign(pred=np.mean(best["val_preds"], axis=0), source="fit_on_train"),
                     S.loc[test, cols].assign(pred=test_pred[test].to_numpy(), source=args.mode)],
                    ignore_index=True)
    finish(folder, name, out, tuning, cfg,
           extra={"mode": args.mode, "device": str(dev), "lookback": panel.lookback, "target": target,
                  "epochs_walk_forward": n_epochs, "seeds": args.seeds, "retrain_every": args.retrain_every,
                  "window_years": args.window_years, "walk_forward": wf_log})
    torch.save({"state_dict": model.state_dict(), "config": cfg, "n_features": panel.n_features,
                "lookback": panel.lookback, "features": panel.meta["features"]},
               folder / "results" / "model_final.pt")
