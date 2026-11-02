# 04 — XGBoost: EDA & feature selection

Prerequisite: `00_data_audit/REPORT.md`.

## Why XGBoost is prepared differently
- **Missing values are kept.** XGBoost learns a default direction for NaN at every split, so imputing would throw away the information that a stock is newly listed / has short history. The table below checks whether missingness relates to the target.
- Stock features are per-date ranks (stable split thresholds across years); market-regime columns are kept raw.
- Only near-exact duplicates are removed first (|corr| ≥ 0.95); boosting then reveals usefulness through SHAP and validation permutation importance. Early stopping on validation guards the selection fit.

## Missingness

| feature               |   na_share_train |   na_share_val |   mean_y_when_na |   mean_y_when_present |
|:----------------------|-----------------:|---------------:|-----------------:|----------------------:|
| mom_60d               |           0.0020 |         0.0000 |           0.0280 |                0.0000 |
| mom_120d              |           0.0048 |         0.0000 |           0.0527 |               -0.0002 |
| mom_252d              |           0.0191 |         0.0000 |           0.0229 |               -0.0003 |
| rel_mom_60d           |           0.0020 |         0.0000 |           0.0280 |                0.0000 |
| rel_mom_120d          |           0.0048 |         0.0000 |           0.0527 |               -0.0002 |
| vol_60d               |           0.0020 |         0.0000 |           0.0280 |                0.0000 |
| vol_ratio_20_60       |           0.0020 |         0.0000 |           0.0280 |                0.0000 |
| vol_adj_mom_60d       |           0.0020 |         0.0000 |           0.0280 |                0.0000 |
| mom_accel_20d         |           0.0012 |         0.0000 |           0.0409 |                0.0001 |
| dist_high_60d         |           0.0019 |         0.0000 |           0.0275 |                0.0000 |
| dist_low_60d          |           0.0019 |         0.0000 |           0.0275 |                0.0000 |
| dist_ma50             |           0.0016 |         0.0000 |           0.0348 |                0.0000 |
| dist_ma200            |           0.0094 |         0.0000 |           0.0310 |               -0.0002 |
| ma50_ma200_ratio      |           0.0094 |         0.0000 |           0.0310 |               -0.0002 |
| ma50_slope_20d        |           0.0024 |         0.0000 |           0.0378 |                0.0000 |
| trend_consistency_60d |           0.0020 |         0.0000 |           0.0280 |                0.0000 |
| volume_trend_20_60    |           0.0019 |         0.0000 |           0.0275 |                0.0000 |
| clv_mean_20d          |           0.0024 |         0.0015 |           0.0541 |               -0.0000 |
| beta_60d              |           0.0020 |         0.0000 |           0.0280 |                0.0000 |
| beta_120d             |           0.0048 |         0.0000 |           0.0527 |               -0.0002 |
| corr_60d              |           0.0020 |         0.0000 |           0.0280 |                0.0000 |
| rel_vol_60d           |           0.0020 |         0.0000 |           0.0280 |                0.0000 |
| resid_ret_60d         |           0.0048 |         0.0000 |           0.0527 |               -0.0002 |
| variance_ratio_5_60   |           0.0016 |         0.0000 |           0.0348 |                0.0000 |
| ret_autocorr_60d      |           0.0014 |         0.0000 |           0.0377 |                0.0000 |
| bench_drawdown_252d   |           0.0050 |         0.0000 |           0.0001 |                0.0001 |

![missing](figures/04_missing_train.png)

## Selection

1. Duplicates removed → 51 (43 stock + 8 market).
2. XGBoost (depth 4, lr 0.03, early-stopped at 105 trees) on every 5th train date. Features that make up 95% of total mean |SHAP| on validation: 39.
3. Removed features whose within-date shuffle *raises* validation IC by more than one s.d.: `mfi_14`, `price_volume_interaction`, `rel_volume_20d`, `volume_trend_20_60`, `trend_consistency_60d`, `mom_accel_20d`, `vol_ret_corr_20d`, `mom_120d`, `log_volume`, `beta_60d`, `ret_5d`, `overnight_gap_mean_20d`, `corr_60d`, `dist_ma50`, `mom_20d`, `gap_vol_20d`.
→ **28 features**: `ret_1d`, `mom_252d`, `vol_60d`, `downside_vol_20d`, `vol_ratio_20_60`, `vol_adj_mom_60d`, `dist_high_20d`, `dist_high_60d`, `dist_low_60d`, `dist_ma200`, `ma50_ma200_ratio`, `ma50_slope_20d`, `log_traded_value_med_20d`, `intraday_return`, `intraday_range`, `clv_mean_20d`, `beta_120d`, `resid_ret_60d`, `yz_vol_20d`, `variance_ratio_5_60`, `adx_14`, `r_b`, `bench_mom_60d`, `bench_mom_120d`, `bench_vol_20d`, `bench_vol_60d`, `bench_dist_ma200`, `bench_drawdown_252d`

![shap](figures/01_shap_importance.png)
![perm](figures/02_permutation_importance.png)
![dependence](figures/03_shap_dependence.png)

|                          |   mean_abs_shap |   cum_share |   total_gain |    ic_drop |   ic_drop_sd |     ic_mean |   pdf_tier | selected   |
|:-------------------------|----------------:|------------:|-------------:|-----------:|-------------:|------------:|-----------:|:-----------|
| adx_14                   |       0.0004798 |      0.7002 |       7.968  |  6.387e-05 |    0.0003548 |   0.02315   |          3 | True       |
| bench_dist_ma200         |       0.0002697 |      0.8772 |      11.56   |  0         |    0         | nan         |        nan | True       |
| bench_drawdown_252d      |       0.0001778 |      0.9554 |       8.082  |  0         |    0         | nan         |        nan | True       |
| bench_mom_120d           |       0.0004324 |      0.7325 |      30.98   |  0         |    0         | nan         |        nan | True       |
| bench_mom_60d            |       0.0004221 |      0.7789 |      22.08   |  0         |    0         | nan         |        nan | True       |
| bench_ret_20d            |       9.664e-05 |      0.9914 |       7.842  |  0         |    0         | nan         |        nan | False      |
| bench_vol_20d            |       0.0002847 |      0.8674 |       8.449  |  0         |    0         | nan         |        nan | True       |
| bench_vol_60d            |       0.0005584 |      0.6268 |      42.15   |  0         |    0         | nan         |        nan | True       |
| beta_120d                |       0.0005194 |      0.6648 |       9.939  |  0.005947  |    0.0007071 |  -0.01931   |          2 | True       |
| beta_60d                 |       0.0001324 |      0.971  |       4.391  | -0.0008949 |    0.0001716 |  -0.0317    |          2 | False      |
| clv_mean_20d             |       0.0003985 |      0.7933 |       7.179  |  0.0001378 |    5.999e-05 |  -7.166e-05 |          3 | True       |
| cmf_20                   |       0.0001231 |      0.9755 |       2.798  | -3.738e-05 |    8.354e-05 |  -0.005856  |          3 | False      |
| corr_60d                 |       0.0005278 |      0.646  |       7.698  | -0.001218  |    5.872e-05 |  -0.0317    |          2 | False      |
| dist_high_20d            |       0.0006353 |      0.5619 |       5.827  |  0.0002734 |    0.0006075 |  -0.002544  |          4 | True       |
| dist_high_60d            |       0.0003732 |      0.8349 |       9.013  |  0.001179  |    0.0004574 |  -0.007791  |          4 | True       |
| dist_low_60d             |       0.0008223 |      0.49   |       9.367  |  0.0001036 |    0.0009554 |  -0.002023  |          4 | True       |
| dist_ma20                |       6.028e-05 |      0.9963 |       1.391  | -2.689e-05 |    0.0001168 |  -0.0104    |          2 | False      |
| dist_ma200               |       0.001329  |      0.4121 |      13.13   |  0.009362  |    0.002582  |   0.03801   |          2 | True       |
| dist_ma50                |       0.0007041 |      0.5156 |       4.841  | -0.001791  |    0.0004351 |  -0.01153   |          2 | False      |
| downside_vol_20d         |       0.0002215 |      0.9208 |       4.157  |  0.0006925 |    0.0004293 |  -0.006286  |          2 | True       |
| gap_vol_20d              |       0.001322  |      0.4601 |       8.466  | -0.00213   |    0.0005416 |  -0.008822  |          3 | False      |
| intraday_range           |       0.0002532 |      0.8864 |       3.267  |  0.0004169 |    0.0001509 |   0.0002964 |          4 | True       |
| intraday_return          |       0.0001972 |      0.9352 |       1.818  |  9.146e-05 |    0.0001457 |  -0.005726  |          3 | True       |
| log_traded_value_med_20d |       0.005727  |      0.208  |      52.56   |  0.03615   |    0.002375  |  -0.04159   |          4 | True       |
| log_volume               |       0.0003803 |      0.8213 |       9.572  | -0.0008728 |    0.0004025 |  -0.03832   |          4 | False      |
| ma50_ma200_ratio         |       0.002357  |      0.2936 |      33.95   |  0.01801   |    0.001398  |   0.05204   |          4 | True       |
| ma50_slope_20d           |       0.0003905 |      0.8075 |       7.905  |  0.000422  |    0.0001398 |   0.02349   |          4 | True       |
| mfi_14                   |       5.269e-05 |      0.9982 |       1.013  | -0.0001907 |    6.203e-05 |  -0.009325  |          3 | False      |
| mom_120d                 |       0.0003128 |      0.8462 |       7.259  | -0.0006488 |    0.0004374 |   0.03307   |          3 | False      |
| mom_20d                  |       0.0002526 |      0.8956 |       3.215  | -0.001984  |    0.0001081 |  -0.0123    |          3 | False      |
| mom_252d                 |       0.001933  |      0.3638 |      31.81   |  0.008038  |    0.0009871 |   0.05029   |          3 | True       |
| mom_accel_20d            |       0.0002432 |      0.9044 |       3.738  | -0.0005062 |    7.748e-05 |  -0.009146  |          2 | False      |
| overnight_gap_mean_20d   |       0.0006406 |      0.5388 |       7.931  | -0.001134  |    0.0003037 |  -0.008877  |          3 | False      |
| overnight_ret            |       0.0001185 |      0.9798 |       0.5986 |  0.0002092 |    0.000112  |   0.002689  |          3 | False      |
| price_volume_interaction |       0.0001936 |      0.9422 |       2.349  | -0.0002061 |    7.15e-05  |  -0.007679  |          3 | False      |
| r_b                      |       0.0002289 |      0.9127 |       5.55   |  0         |    0         | nan         |        nan | True       |
| rel_volume_20d           |       0.0001119 |      0.9839 |       1.893  | -0.0002472 |    0.0001049 |  -0.002813  |          2 | False      |
| resid_ret_60d            |       0.0004943 |      0.6828 |       8.001  |  0.001328  |    0.0004145 |  -0.01669   |          1 | True       |
| ret_1d                   |       0.0004557 |      0.7168 |       3.333  |  3.846e-05 |    0.0003449 |  -0.004967  |          2 | True       |
| ret_5d                   |       0.0002983 |      0.8571 |       2.304  | -0.0009775 |    0.0001147 |  -0.01104   |          3 | False      |
| ret_autocorr_60d         |       0.0001455 |      0.9662 |       4.205  | -2.079e-05 |    0.0004298 |  -0.004788  |          3 | False      |
| rsi_14                   |       5.038e-05 |      1      |       1.025  | -3.484e-05 |    3.82e-05  |  -0.007726  |          2 | False      |
| trend_consistency_60d    |       0.0001527 |      0.9609 |       3.361  | -0.0004797 |    0.0002386 |   0.01343   |          4 | False      |
| variance_ratio_5_60      |       0.0001992 |      0.928  |       4.087  | -0.0001081 |    0.0001303 |   0.00363   |          2 | True       |
| vol_20d                  |       0.000109  |      0.9878 |       2.414  |  0.0004478 |    0.0001221 |  -0.007966  |          2 | False      |
| vol_60d                  |       0.0004319 |      0.7482 |      14.86   |  0.00176   |    0.0003446 |  -0.006366  |          2 | True       |
| vol_adj_mom_60d          |       0.0006054 |      0.6065 |       4.699  |  7.863e-05 |    0.0003079 |   0.01686   |          3 | True       |
| vol_ratio_20_60          |       0.0001856 |      0.9489 |       4.83   | -6.059e-05 |    0.0001236 |  -0.002301  |          2 | True       |
| vol_ret_corr_20d         |       0.0004231 |      0.7635 |       4.703  | -0.0006366 |    0.0004112 |  -0.02166   |          3 | False      |
| volume_trend_20_60       |       7.455e-05 |      0.9941 |       1.757  | -0.0002712 |    3.246e-05 |  -0.005445  |          4 | False      |
| yz_vol_20d               |       0.0006231 |      0.5845 |       9.198  |  0.002117  |    0.0004155 |   0.01018   |          2 | True       |

## Sanity check on validation

| feature set             |   n |   trees (early stop) |   val IC |   val IC NW t |
|:------------------------|----:|---------------------:|---------:|--------------:|
| all 58 (stock + market) |  58 |                   90 |   0.0816 |        3.4337 |
| after duplicates        |  51 |                  105 |   0.0779 |        3.5893 |
| selected                |  28 |                   75 |   0.0917 |        3.7888 |

Caveat: validation is used both for early stopping and for this check, so these numbers are mildly optimistic. The test period stays untouched.

## Model-ready dataset

`data/xgb_dataset.parquet` — 505,146 rows, 28 features (float32, **NaN retained**), targets `y_raw, y_win, y_rank, y_class`.

| split   |   rows |   with_target | first               | last                |
|:--------|-------:|--------------:|:--------------------|:--------------------|
| burn_in |  55579 |         55579 | 2018-01-01 00:00:00 | 2018-12-31 00:00:00 |
| purge   |   9520 |          9520 | 2022-12-05 00:00:00 | 2024-06-28 00:00:00 |
| test    | 126854 |        122094 | 2024-07-01 00:00:00 | 2026-08-21 00:00:00 |
| train   | 230131 |        230131 | 2019-01-01 00:00:00 | 2022-12-02 00:00:00 |
| val     |  83062 |         83062 | 2023-01-02 00:00:00 | 2024-05-30 00:00:00 |

Use rows with `split == 'train'` and `has_target`; early-stop on `val`; evaluate with `y_raw`. `y_rank` with `objective='reg:squarederror'` or `rank:pairwise` grouped by Date are both worth testing.