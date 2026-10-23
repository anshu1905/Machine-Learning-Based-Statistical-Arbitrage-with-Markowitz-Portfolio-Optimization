# Top-10 + Markowitz backtest: ML vs historical expected returns

Test period 2024-07-01 -> 2026-08-21, 27 rebalances every 20 trading days, cost 10 bps per unit of one-way turnover, Markowitz cap 25%, Sharpe with rf = 0.0%. EW = equal weight, MV = Markowitz max-Sharpe. `daily test IC` = mean daily Spearman correlation between the expected-return input (historical 252-day mean, or the model's prediction) and the realised 20-day excess return, over every test day; `IC t` = its Newey-West t-stat. `t vs Hist.` = paired t-stat of the per-period return difference against the historical-mean portfolio with the same weighting.

| source        | weighting   | CAGR   | ann. vol   |   Sharpe | max drawdown   | total return   | avg turnover   | daily test IC   | IC t   | beats Hist. (periods)   | t vs Hist.   |
|:--------------|:------------|:-------|:-----------|---------:|:---------------|:---------------|:---------------|:----------------|:-------|:------------------------|:-------------|
| Historical    | EW          | 9.0%   | 28.4%      |     0.45 | -31.5%         | 20.0%          | 59.9%          | 0.0008          | 0.03   |                         |              |
| Historical    | MV          | 9.5%   | 27.9%      |     0.46 | -29.5%         | 21.0%          | 72.2%          | 0.0008          | 0.03   |                         |              |
| Ridge         | EW          | 20.9%  | 25.4%      |     0.87 | -24.2%         | 49.3%          | 97.9%          | 0.0100          | 0.64   | 51.9%                   | 0.82         |
| Ridge         | MV          | 17.9%  | 24.0%      |     0.81 | -27.1%         | 41.7%          | 103.8%         | 0.0100          | 0.64   | 51.9%                   | 0.60         |
| ElasticNet    | EW          | 32.9%  | 25.7%      |     1.23 | -22.3%         | 82.2%          | 102.2%         | 0.0038          | 0.20   | 44.4%                   | 1.34         |
| ElasticNet    | MV          | 34.2%  | 24.3%      |     1.33 | -18.7%         | 85.9%          | 111.8%         | 0.0038          | 0.20   | 55.6%                   | 1.31         |
| Random Forest | EW          | 25.7%  | 22.8%      |     1.12 | -21.7%         | 62.0%          | 83.3%          | 0.0208          | 0.98   | 51.9%                   | 0.99         |
| Random Forest | MV          | 19.5%  | 22.0%      |     0.92 | -22.0%         | 45.7%          | 93.1%          | 0.0208          | 0.98   | 59.3%                   | 0.68         |
| XGBoost       | EW          | 25.8%  | 23.8%      |     1.08 | -26.0%         | 62.2%          | 110.7%         | 0.0183          | 0.83   | 48.1%                   | 1.06         |
| XGBoost       | MV          | 15.4%  | 22.7%      |     0.74 | -28.1%         | 35.3%          | 108.3%         | 0.0183          | 0.83   | 44.4%                   | 0.44         |
| RNN           | EW          | 29.5%  | 24.7%      |     1.17 | -20.7%         | 72.6%          | 99.6%          | 0.0092          | 0.53   | 59.3%                   | 1.48         |
| RNN           | MV          | 29.0%  | 23.5%      |     1.2  | -18.0%         | 71.2%          | 106.6%         | 0.0092          | 0.53   | 51.9%                   | 1.35         |
| LSTM          | EW          | 26.8%  | 24.3%      |     1.1  | -25.7%         | 65.1%          | 88.2%          | 0.0137          | 0.74   | 55.6%                   | 1.18         |
| LSTM          | MV          | 21.2%  | 23.1%      |     0.95 | -25.2%         | 50.0%          | 92.7%          | 0.0137          | 0.74   | 55.6%                   | 0.83         |
| ARIMA         | EW          | 11.4%  | 27.8%      |     0.53 | -28.4%         | 25.5%          | 65.4%          | 0.0072          | 0.25   | 51.9%                   | 0.58         |
| ARIMA         | MV          | 18.9%  | 25.9%      |     0.8  | -26.7%         | 44.0%          | 93.5%          | 0.0072          | 0.25   | 59.3%                   | 1.26         |
| Benchmark     | -           | 7.6%   | 17.5%      |     0.51 | -20.1%         | 16.7%          |                |                 |        |                         |              |

Markowitz fell back to minimum variance (no positive mu among the picks) in 0 of 216 portfolio-periods.