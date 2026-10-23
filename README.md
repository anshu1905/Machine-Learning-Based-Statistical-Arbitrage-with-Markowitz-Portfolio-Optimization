# ML Statistical Arbitrage + Markowitz — Indian equities

This folder holds the EDA, feature selection, **model-ready datasets** and **training code** for the seven
forecasting models. Training steps, server setup and hardware requirements: [TRAINING.md](TRAINING.md).
The walk-forward Markowitz backtest is the next stage.

```
Implementation/
├── features_by_stock.xlsx      source: 238 stocks × 67 columns, 2018-01 → 2026-08 (Bloomberg)
├── data/                       shared: panel_raw / panel_clean parquet, IC + correlation tables
├── common/                     shared code: config & splits, transforms, IC tools, extra features, plotting
├── 00_data_audit/              shared EDA: integrity, leakage, target, missingness, IC, redundancy
├── 01_ridge/                   ┐
├── 02_elasticnet/              │  each folder: prepare_data.py → REPORT.md, figures/,
├── 03_random_forest/           │  selected_features.json, data/ (model-ready dataset);
├── 04_xgboost/                 │  train.py → results/ (predictions, metrics, model)
├── 05_rnn/                     │  + dataset.py (PyTorch Dataset)
├── 06_lstm/                    │  + dataset.py (PyTorch Dataset)
├── 07_arima/                   ┘  (per-stock series, orders.csv)
├── run_all.sh                  rebuilds every dataset (~5 min)
├── train_all.sh                trains all seven models (--quick for a smoke test)
├── compare_models.py           one comparison table from every results/metrics.json
├── TRAINING.md                 training protocol, per-model steps, GPU server setup, requirements
└── requirements.txt
```

Start with [00_data_audit/REPORT.md](00_data_audit/REPORT.md), then each model's `REPORT.md`.

## Setup

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
```

On macOS XGBoost also needs the OpenMP runtime: `brew install libomp`.

## Decisions every model shares

| item | choice |
|---|---|
| Target | `target_excess_ret_20d` = 20-trading-day forward simple return − benchmark's. The benchmark behaves like the universe average (the cross-sectional mean of the target is ≈ 0 every day). |
| Target variants | `y_raw` (evaluate / P&L), `y_win` (per-date 1/99% winsorized; default training target), `y_rank` (per-date Gaussian rank), `y_class` (outperform 0/1) |
| Splits | burn-in 2018 · **train 2019 → 2022** · **val 2023 → Jun 2024** · **test Jul 2024 → Aug 2026**, with a 20-trading-day **purge** before val and before test |
| Leakage | `fwd_ret_20d`, `bench_fwd_ret`, `target_*` are never features. All selection statistics use train only (plus val for the model-based checks, which is stated where it happens). |
| Scaling | Per-date (cross-sectional) transforms: they use same-day data only, so they need no fitting and carry over to walk-forward unchanged. The only fitted scaler (market columns in RNN/LSTM) uses train statistics. |
| IC t-stats | Newey–West with 20 lags, because daily 20-day targets overlap |
| Extra features | 10 single-stock features from `statarb_features_ranked.pdf` that the file lacked (Amihud, Yang–Zhang vol, variance ratio, RSI, ADX, MFI, CMF, …) |

## Per-model preparation at a glance

| model | inputs | transform | selection method | features | quick val IC* |
|---|---|---|---|---|---|
| Ridge | stock only | per-date winsorize + z-score | IC screen → decorrelate (0.8) → VIF < 5 | 20 | 0.105 |
| ElasticNet | stock only | per-date winsorize + z-score | remove near-duplicates (0.9); stability selection as diagnostic | 39 (13 stable) | 0.089 |
| Random Forest | stock + market regime | per-date rank; market raw | Boruta-style shadows + validation permutation IC | 19 (12 + 7 market) | 0.083 |
| XGBoost | stock + market regime | per-date rank, **NaN kept** | mean \|SHAP\| (95%) + validation permutation IC | 27 (20 + 7 market) | 0.089 |
| RNN | 20-day windows | per-date Gaussian rank; market train z-score | 8 IC core + 4 fast daily + 3 market | 15 × 20 steps | — |
| LSTM | 60-day windows | as RNN | 14 IC core + 3 fast daily + 5 market | 22 × 60 steps | — |
| ARIMA | own past only | daily log excess return | ADF/KPSS → d = 0; ACF/PACF, Ljung–Box, BIC grid for (p, q) | orders per stock | — |

\* Mean daily rank IC on validation from an *untuned* default model, used only to sanity-check each feature set.
Newey–West t ≈ 3.4–4.0. Validation spans ~17 independent 20-day periods, so the standard error is ~0.03:
these numbers do not rank the models.

## Main EDA findings

- The signal is weak but real. The best single features have train IC ≈ 0.04–0.05 (NW t ≈ 2–2.7). They are size/liquidity
  (smaller, less liquid names outperform: `log_traded_value_med_20d`, `amihud_log_20d`) and long-horizon trend
  (`mom_252d`, `ma50_ma200_ratio`, `dist_ma200`). Short-term reversal is weak.
- Feature relevance shifts with market volatility (see `03_random_forest/figures/04_regime_ic.png`). This is why the
  tree and sequence models get market-regime inputs.
- Adding lagged feature values does not raise validation IC linearly (RNN/LSTM reports). Any RNN/LSTM gain must come
  from non-linear temporal patterns.
- Daily excess returns are close to white noise: BIC picks ARIMA(0,0,0) for 76% of stocks.
- Data caveats: unadjusted demergers (Siemens 2025-04, Tata Motors 2025-10) and survivorship (today's universe
  back-filled) → winsorized targets for training; results will be somewhat optimistic.

## Loading a dataset

```python
import pandas as pd, json
df = pd.read_parquet("01_ridge/data/ridge_dataset.parquet")
feats = json.load(open("01_ridge/selected_features.json"))["features"]
tr = df[(df.split == "train") & df.has_target]
X, y = tr[feats], tr.y_win
```
