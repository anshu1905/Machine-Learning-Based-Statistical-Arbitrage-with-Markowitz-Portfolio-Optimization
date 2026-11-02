"""Ready-to-use PyTorch datasets for this model folder.

    from dataset import load
    train_ds, val_ds, test_ds = load(target="y_win")
    loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.torch_dataset import SequenceDataset  # noqa: E402

DATA = HERE / "data"


def load(target="y_win", with_target_only=True):
    return tuple(SequenceDataset(DATA, s, target=target, with_target_only=with_target_only)
                 for s in ("train", "val", "test"))


if __name__ == "__main__":
    for name, ds in zip(("train", "val", "test"), load()):
        x, y = ds[0]
        print(f"{name}: {len(ds):,} samples, x {tuple(x.shape)}, y {float(y):+.4f}")
