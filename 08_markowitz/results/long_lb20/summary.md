# Top-10 + Markowitz backtest (long-only, 20-day lookback): ML vs historical expected returns

Test period 2024-07-01 -> 2026-08-21, 27 rebalances every 20 trading days, cost 10 bps per unit of one-way turnover, Markowitz cap 25%, Sharpe with rf = 0.0%. EW = equal weight, MV = Markowitz max-Sharpe. `daily test IC` = mean daily Spearman correlation between the expected-return input (historical 20-day mean, or the model's prediction) and the realised 20-day excess return, over every test day; `IC t` = its Newey-West t-stat. `t vs Hist.` = paired t-stat of the per-period return difference against the historical-mean portfolio with the same weighting.

| source        | weighting   | CAGR   | ann. vol   |   Sharpe | max drawdown   | total return   | avg turnover   | daily test IC   | IC t   | beats Hist. (periods)   | t vs Hist.   |
|:--------------|:------------|:-------|:-----------|---------:|:---------------|:---------------|:---------------|:----------------|:-------|:------------------------|:-------------|
| Historical    | EW          | 23.0%  | 24.5%      |     0.97 | -24.6%         | 54.7%          | 179.2%         | 0.0052          | 0.26   |                         |              |
| Historical    | MV          | 23.9%  | 24.3%      |     1.01 | -22.9%         | 57.3%          | 186.8%         | 0.0052          | 0.26   |                         |              |
| Ridge         | EW          | 20.9%  | 25.4%      |     0.87 | -24.2%         | 49.3%          | 97.9%          | 0.0100          | 0.64   | 44.4%                   | -0.25        |
| Ridge         | MV          | 21.9%  | 24.9%      |     0.92 | -28.7%         | 51.9%          | 122.5%         | 0.0100          | 0.64   | 48.1%                   | -0.21        |
| ElasticNet    | EW          | 32.9%  | 25.7%      |     1.23 | -22.3%         | 82.2%          | 102.2%         | 0.0038          | 0.20   | 48.1%                   | 0.52         |
| ElasticNet    | MV          | 43.8%  | 25.4%      |     1.56 | -17.7%         | 115.4%         | 124.5%         | 0.0038          | 0.20   | 51.9%                   | 0.85         |
| Random Forest | EW          | 25.7%  | 22.8%      |     1.12 | -21.7%         | 62.0%          | 83.3%          | 0.0208          | 0.98   | 63.0%                   | 0.06         |
| Random Forest | MV          | 26.9%  | 23.1%      |     1.15 | -24.4%         | 65.2%          | 107.1%         | 0.0208          | 0.98   | 48.1%                   | 0.04         |
| XGBoost       | EW          | 25.8%  | 23.8%      |     1.08 | -26.0%         | 62.2%          | 110.7%         | 0.0183          | 0.83   | 51.9%                   | 0.12         |
| XGBoost       | MV          | 27.6%  | 23.6%      |     1.15 | -27.1%         | 67.4%          | 127.1%         | 0.0183          | 0.83   | 55.6%                   | 0.11         |
| RNN           | EW          | 29.5%  | 24.7%      |     1.17 | -20.7%         | 72.6%          | 99.6%          | 0.0092          | 0.53   | 55.6%                   | 0.31         |
| RNN           | MV          | 37.9%  | 24.3%      |     1.44 | -19.2%         | 97.2%          | 122.0%         | 0.0092          | 0.53   | 55.6%                   | 0.61         |
| LSTM          | EW          | 26.8%  | 24.3%      |     1.1  | -25.7%         | 65.1%          | 88.2%          | 0.0137          | 0.74   | 59.3%                   | 0.17         |
| LSTM          | MV          | 37.0%  | 24.2%      |     1.42 | -23.5%         | 94.5%          | 113.0%         | 0.0137          | 0.74   | 55.6%                   | 0.57         |
| ARIMA         | EW          | 11.4%  | 27.8%      |     0.53 | -28.4%         | 25.5%          | 65.4%          | 0.0072          | 0.25   | 44.4%                   | -0.80        |
| ARIMA         | MV          | 9.7%   | 26.6%      |     0.48 | -24.1%         | 21.7%          | 109.3%         | 0.0072          | 0.25   | 44.4%                   | -0.96        |
| Benchmark     | -           | 7.6%   | 17.5%      |     0.51 | -20.1%         | 16.7%          |                |                 |        |                         |              |

Markowitz fell back to minimum variance (no positive mu among the picks) in 57 of 216 portfolio-periods.