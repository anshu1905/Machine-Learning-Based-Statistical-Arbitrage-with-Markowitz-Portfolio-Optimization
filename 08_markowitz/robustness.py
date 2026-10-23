"""Re-run backtest.py for all 20 possible start days of the 20-day rebalance cycle.

With only ~27 rebalances, one start day can be lucky or unlucky. This shows whether the
ML-vs-historical ranking holds for every start day. Extra arguments go to backtest.py.

    python 08_markowitz/robustness.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    procs = [subprocess.Popen([sys.executable, str(HERE / "backtest.py"), "--offset", str(o), "--out", f"{tmp}/{o}",
                               *sys.argv[1:]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for o in range(20)]
    if any(p.wait() for p in procs):
        sys.exit("a backtest run failed; run backtest.py --offset N directly to see the error")
    d = pd.concat([pd.read_csv(f"{tmp}/{o}/summary.csv").assign(offset=o) for o in range(20)])

d["strategy"] = d.source + " " + d.weighting
sharpe = d.pivot(index="offset", columns="strategy", values="Sharpe")
cagr = d.pivot(index="offset", columns="strategy", values="CAGR")
out = pd.DataFrame({"mean Sharpe": sharpe.mean(), "min Sharpe": sharpe.min(), "max Sharpe": sharpe.max(),
                    "mean CAGR": cagr.mean()})
for wt in ("EW", "MV"):
    cols = [c for c in sharpe if c.endswith(wt)]
    out.loc[cols, "Sharpe > Historical (share of start days)"] = sharpe[cols].gt(sharpe[f"Historical {wt}"], axis=0).mean()
out = out.sort_values("mean Sharpe", ascending=False)
out.to_csv(HERE / "results/offset_robustness.csv")
(HERE / "results/offset_robustness.md").write_text(
    "# Robustness: all 20 start days of the 20-day rebalance cycle\n\n" + out.to_markdown(floatfmt=".3f"))
print(out.round(3).to_string())
