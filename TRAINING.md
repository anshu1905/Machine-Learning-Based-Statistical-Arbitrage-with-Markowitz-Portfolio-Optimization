# Model training — steps and requirements

Every model has a `train.py` in its folder. All seven follow the same protocol and write the same outputs, so they
can be compared directly and fed to the Markowitz stage in the same way.

## 1. The common protocol

| step | what happens | data used |
|---|---|---|
| **1. Tune** | fit every hyper-parameter config, score it by **mean daily rank IC** (Spearman correlation of prediction vs realised 20d excess return, per date) | fit on **train** 2019 → Dec 2022, score on **validation** Jan 2023 → May 2024 |
| **2. Walk-forward test** | with the best config, refit every `--retrain-every` trading days (default 63 ≈ quarterly) and predict until the next refit | each refit at date T uses every row whose 20-day target ended before T (expanding window from 2019, or rolling with `--window-years`) |
| **3. Save** | predictions, metrics, tuning table, final model | `results/` inside the model folder |

No look-ahead: a refit at date T only sees rows dated ≤ T − 21 trading days, so every training target has fully
realised before T. Test data (Jul 2024 → Aug 2026) is never used for any choice.

`--mode fixed` replaces step 2 with a single refit on train + validation, then predicts all of test (faster, less realistic).

### Outputs (`<model>/results/`)

| file | content |
|---|---|
| `predictions.parquet` | `Date, Symbol, split, tradable, y_raw, pred, pred_z, source`, for validation (fit on train) and test (walk-forward). `pred_z` is the per-date Gaussian rank of `pred`, i.e. the z<sub>i</sub> in μ<sub>i</sub> = IC · σ<sub>i</sub> · z<sub>i</sub> for Markowitz |
| `metrics.json` | validation and test: mean IC, Newey–West t, ICIR, IC hit rate, top-minus-bottom quintile return, IC by year; best parameters; walk-forward log |
| `tuning.csv` | every config tried with its validation IC |
| `cumulative_ic.png` | cumulative daily IC over validation and test |
| `model_final.*` | the model from the last walk-forward refit (`.joblib`, `.json` for XGBoost, `.pt` for RNN/LSTM) |

After all models have run, `compare_models.py` writes `results_summary.md` with one row per model.

## 2. Setting up the GPU server

**Software:** Linux x86-64, an NVIDIA driver with CUDA 12 support (`nvidia-smi` works), Python 3.11+.

```bash
# from your laptop: copy the project (≈ 350 MB without the venv, the Excel file and the raw parquet)
rsync -av --exclude .venv --exclude features_by_stock.xlsx --exclude data/panel_raw.parquet \
  "Implementation/" user@server:~/statarb/
```

```bash
# on the server
cd ~/statarb
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The Linux wheels of `torch` and `xgboost` on PyPI already include CUDA. Check that both see the GPU:

```bash
.venv/bin/python -c "import torch, xgboost; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), xgboost.build_info()['USE_CUDA'])"
```

Expected output: `True <your GPU name> True`. If torch prints `False`, the driver is too old for the bundled CUDA
runtime. Install the torch build that matches the server's CUDA version from pytorch.org instead.

Run inside `tmux` (or with `nohup`) so a dropped SSH session doesn't kill training:

```bash
tmux new -s train
./train_all.sh --quick      # 2-5 min smoke test of all seven; check for errors first
./train_all.sh              # full run; logs in logs/<model>.log, summary in results_summary.md
```

Use `CUDA_VISIBLE_DEVICES=1 ./train_all.sh` to pick a specific GPU. Models are independent, so they can also be run
one by one or in parallel on different GPUs/terminals.

## 3. Requirements at a glance

Runtimes are rough estimates scaled from the local smoke tests (10-core Apple M-series laptop) and should be treated
as orders of magnitude.

| model | runs on | GPU used? | RAM | GPU memory | rough full-run time |
|---|---|---|---|---|---|
| Ridge | CPU | no | 4 GB | – | < 1 min |
| ElasticNet | CPU | no | 4 GB | – | 2–5 min |
| Random Forest | CPU (all cores) | no — scikit-learn has no GPU forest | 8 GB | – | 10–30 min (scales with cores) |
| XGBoost | **GPU** (CPU fallback) | yes, `device="cuda"` | 8 GB | ≥ 2 GB | GPU 5–15 min · CPU 30–60 min |
| RNN | **GPU** (MPS/CPU fallback) | yes, CUDA + fp16 | 8 GB | ≥ 2 GB | GPU 10–30 min |
| LSTM | **GPU** (MPS/CPU fallback) | yes, CUDA + fp16 | 8 GB | ≥ 4 GB | GPU 30–90 min |
| ARIMA | CPU (parallel per stock) | no | 4 GB | – | 5–15 min (`--test-step 5`) |

A server with ≥ 8 CPU cores, 16 GB RAM and any recent NVIDIA GPU (≥ 8 GB, e.g. T4 / A10 / RTX 30-40 series) is
comfortably enough. The GPU only matters for XGBoost, RNN and LSTM.

Common flags (all tabular models and RNN/LSTM): `--mode wf|fixed`, `--retrain-every 21|63|126`, `--window-years 3`
(rolling instead of expanding), `--n-iter N` (random subset of the grid), `--quick` (smoke test).

---

## 4. Per-model steps

### 01 Ridge Regression — `01_ridge/train.py`

**Input:** `data/ridge_dataset.parquet`, 20 features (per-date winsorized z-scores, no NaN).

1. Tune the grid on train → validation: `alpha` ∈ {0.01 … 10⁵} (8 values) × target ∈ {`y_win`, `y_rank`} = 16 fits.
2. Pick the highest validation IC.
3. Walk-forward over test: refit quarterly on all realised data and predict the next quarter.
4. Save predictions, metrics, coefficients of the last refit (`metrics.json → coefficients`) and the model.

```bash
.venv/bin/python 01_ridge/train.py
```

What to look at: coefficient signs should match the IC signs in `01_ridge/REPORT.md`. Ridge is the baseline every
other model must beat.

### 02 ElasticNet — `02_elasticnet/train.py`

**Input:** `data/elasticnet_dataset.parquet`, 39 features. The L1 part selects among them.

1. The target is divided by its train standard deviation, so one alpha grid fits both `y_win` and `y_rank`.
2. Tune `alpha` ∈ 10⁻⁴…10⁻¹ (7) × `l1_ratio` ∈ {0.1, 0.5, 0.9} × target (2) = 42 fits; record the non-zero coefficient count.
3. Pick the highest validation IC, then refit quarterly over test.
4. Save; `metrics.json → nonzero_coefficients_last_refit` lists the features that survived.

```bash
.venv/bin/python 02_elasticnet/train.py
```

What to look at: whether it keeps roughly the 13 "stable" features from `02_elasticnet/REPORT.md`.

### 03 Random Forest — `03_random_forest/train.py`

**Input:** `data/rf_dataset.parquet`, 19 features (12 stock ranks + 7 market-regime columns).

1. Train on **every 5th date** (`--thin 5`): consecutive 20-day targets overlap 95%, so the extra rows add almost no
   information but make leaves over-confident.
2. Tune `max_depth` {6, 10, 14} × `min_samples_leaf` {100, 500, 2000} × `max_features` {0.33, 0.6} = 18 fits, 300 trees each.
3. Refit the best config with **500 trees** quarterly over test.
4. Save; `metrics.json → impurity_importance_last_refit`.

```bash
.venv/bin/python 03_random_forest/train.py --n-jobs -1    # all cores
```

Requirement: CPU cores. Runtime scales almost linearly with cores. Don't trust the built-in out-of-bag score
(overlapping targets inflate it); the validation IC is the honest score.

### 04 XGBoost — `04_xgboost/train.py`

**Input:** `data/xgb_dataset.parquet`, 27 features, **NaNs kept** (XGBoost learns where missing values go).

1. Detect the device (`--device auto` → CUDA if available).
2. Tune 24 random configs from 96 (`--n-iter 96` runs all): objective {`reg` = squared error on `y_win`, `rank` =
   `rank:pairwise` on per-date decile labels} × depth {3, 4, 6} × learning rate {0.01, 0.03} × min_child_weight
   {100, 500} × colsample {0.5, 0.8} × λ {1, 10}. Each fit uses up to 3000 trees with **early stopping** (100 rounds)
   on validation.
3. Walk-forward with the best config and its early-stopped tree count; no early stopping inside the walk-forward.
4. Save; `metrics.json → gain_importance_last_refit`, `model_final.json`.

```bash
.venv/bin/python 04_xgboost/train.py --device cuda
```

Note: validation is used both for early stopping and for picking the config, so the validation IC is mildly
optimistic. The test IC is the honest number.

### 05 RNN — `05_rnn/train.py`

**Input:** `05_rnn/data/` panel: 15 channels × **20-day windows** (Gaussian ranks; market z-scores fitted on train).

1. Load the whole panel onto the GPU; windows are gathered by indexing (no DataLoader bottleneck).
2. Batches = 8 random dates × a fixed number of random stocks per date (fixed shapes, every date weighted equally).
3. Tune 12 random configs of `hidden` {32, 64} × `layers` {1, 2} × `dropout` {0.1, 0.3} × `lr` {1e-3, 3e-4} ×
   `loss` {`mse` on `y_win`, `ic` = minus per-date correlation with `y_rank`}. AdamW, gradient clipping at 1.0, up to
   40 epochs, **early stopping on validation IC** (patience 6).
4. Retrain the best config with 2 more seeds (3 in total) to get a seed-averaged validation prediction and the
   average best epoch count.
5. Walk-forward: at each quarterly refit, train 3 seeds for that many epochs and **average their predictions**.
6. Save; `model_final.pt` holds the state dict, config, feature list and lookback.

```bash
.venv/bin/python 05_rnn/train.py --device cuda
```

### 06 LSTM — `06_lstm/train.py`

**Input:** `06_lstm/data/` panel: 22 channels × **60-day windows**.

Same steps as the RNN, with a larger grid (`hidden` {32, 64, 128} → 48 configs, 12 sampled). Mixed precision (fp16)
on CUDA. This is the most expensive model; `--n-iter` and `--seeds` set the cost:

```bash
.venv/bin/python 06_lstm/train.py --device cuda                         # 12 configs, 3 seeds
.venv/bin/python 06_lstm/train.py --device cuda --n-iter 24 --seeds 5   # more thorough
```

What to look at for both networks: does the walk-forward test IC beat Ridge's? The EDA linear probe found that
feature history adds no *linear* information, so any gain has to come from non-linear temporal patterns.

### 07 ARIMA — `07_arima/train.py`

**Input:** `07_arima/data/` daily log excess returns per stock, BIC/AIC orders in `orders.csv`.

1. At each rebalance date and for each of the 238 stocks: fit ARIMA(p, 0, q) with a constant on the last `window`
   days, forecast 20 steps, sum → predicted 20-day excess return. ARIMA(0,0,0) uses the closed form (20 × mean).
   Each stock's fit warm-starts from its previous parameters.
2. Tune on validation, rebalancing every 20th day: order rule {BIC orders, AIC orders, fixed (1,0,1)} × window
   {250, 500, 1000} = 9 variants.
3. Test with the best variant, rebalancing every 5th day (`--test-step 1` for daily, ~5× slower).
4. Save predictions (rebalance dates only), metrics (Newey–West lags adjusted for the spacing) and `best_config.json`.

```bash
.venv/bin/python 07_arima/train.py --n-jobs -1
```

Requirement: CPU cores only. Expect ARIMA to be the weakest model: 76% of stocks are white noise under BIC.

---

## 5. After training

1. `.venv/bin/python compare_models.py` → `results_summary.md`. Compare on **test IC and its t-stat**, not validation.
2. Copy the `results/` folders back to your laptop for the Markowitz stage (run inside `Implementation/`):
   ```bash
   rsync -av --include='*/' --include='0*/results/***' --include='results_summary.md' --exclude='*' user@server:~/statarb/ ./
   ```
3. The Markowitz backtest reads `results/predictions.parquet` from each model and turns `pred_z` into expected
   returns (μ<sub>i</sub> = IC · σ<sub>i</sub> · z<sub>i</sub>).
