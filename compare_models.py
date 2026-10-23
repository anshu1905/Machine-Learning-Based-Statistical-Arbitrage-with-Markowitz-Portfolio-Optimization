"""Collect every model's results/metrics.json into one comparison table.

    .venv/bin/python compare_models.py      # prints the table and writes results_summary.md
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
rows = []
for f in sorted(ROOT.glob("0*/results/metrics.json")):
    m = json.loads(f.read_text())
    v, t = m.get("val", {}), m.get("test", {})
    rows.append({
        "model": m["model"], "mode": m.get("mode", "rolling"),
        "val IC": v.get("ic_mean"), "val t": v.get("ic_tstat_nw"),
        "test IC": t.get("ic_mean"), "test t": t.get("ic_tstat_nw"), "test ICIR": t.get("icir_ann"),
        "test IC>0 days": t.get("ic_hit_rate"), "test Q5-Q1 /20d": t.get("q5_minus_q1_20d"),
        "test Q5-Q1 ann. (gross)": t.get("q5_minus_q1_ann_gross"), "test dates": t.get("n_dates"),
    })
table = pd.DataFrame(rows).sort_values("test IC", ascending=False)
md = ["# Model comparison (out-of-sample)", "",
      "IC = mean daily Spearman rank correlation between prediction and realised 20d excess return. "
      "t = Newey-West t-stat. Q5-Q1 = top-quintile minus bottom-quintile realised return, before costs. "
      "Validation numbers were used for tuning; **test is the honest number**.", "",
      table.to_markdown(index=False, floatfmt=".4f")]
(ROOT / "results_summary.md").write_text("\n".join(md))
print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
