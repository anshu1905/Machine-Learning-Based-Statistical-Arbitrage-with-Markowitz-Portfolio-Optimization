"""PyTorch Dataset over the panel arrays written by common/sequence.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Returns (x, y): x = [lookback, n_features] float32, y = scalar target.

    data_dir : folder with X_panel.npy, targets.npz, samples.parquet, meta.json
    split    : "train" | "val" | "test"
    target   : "y_win" (default training target) | "y_raw" | "y_rank" | "y_class"
    with_target_only : drop live rows that have no label yet
    """

    def __init__(self, data_dir, split, target="y_win", lookback=None, with_target_only=True):
        d = Path(data_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.lookback = lookback or self.meta["lookback"]
        self.X = torch.from_numpy(np.load(d / "X_panel.npy"))
        self.Y = torch.from_numpy(np.load(d / "targets.npz")[target])
        idx = pd.read_parquet(d / "samples.parquet")
        idx = idx[idx.split == split]
        if with_target_only:
            idx = idx[idx.has_target]
        self.index = idx.reset_index(drop=True)
        self._s = torch.as_tensor(self.index.s_idx.to_numpy().copy())
        self._t = torch.as_tensor(self.index.t_idx.to_numpy().copy())

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, t = self._s[i], self._t[i]
        return self.X[s, t - self.lookback + 1: t + 1], self.Y[s, t]

    def dates(self):
        """Date of every sample (for per-date IC / cross-sectional evaluation)."""
        return self.index.Date.to_numpy()
