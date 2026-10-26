# Top-10 + Markowitz backtest (long top-10 / short bottom-10, dollar-neutral, 20-day lookback): ML vs historical expected returns

Test period 2024-07-01 -> 2026-08-21, 27 rebalances every 20 trading days, cost 10 bps per unit of one-way turnover, Markowitz cap 25%, Sharpe with rf = 0.0%. EW = equal weight, MV = Markowitz max-Sharpe. `daily test IC` = mean daily Spearman correlation between the expected-return input (historical 20-day mean, or the model's prediction) and the realised 20-day excess return, over every test day; `IC t` = its Newey-West t-stat. `t vs Hist.` = paired t-stat of the per-period return difference against the historical-mean portfolio with the same weighting.

| source        | weighting   | CAGR   | ann. vol   |   Sharpe | max drawdown   | total return   | avg turnover   | daily test IC   | IC t   | beats Hist. (periods)   | t vs Hist.   |
|:--------------|:------------|:-------|:-----------|---------:|:---------------|:---------------|:---------------|:----------------|:-------|:------------------------|:-------------|
| Historical    | EW          | 9.0%   | 23.2%      |     0.49 | -20.2%         | 19.9%          | 370.2%         | 0.0052          | 0.26   |                         |              |
| Historical    | MV          | 10.8%  | 23.4%      |     0.56 | -24.6%         | 24.2%          | 379.3%         | 0.0052          | 0.26   |                         |              |
| Ridge         | EW          | 17.2%  | 20.7%      |     0.87 | -16.8%         | 39.7%          | 164.4%         | 0.0100          | 0.64   | 51.9%                   | 0.27         |
| Ridge         | MV          | 22.4%  | 22.3%      |     1.02 | -21.3%         | 53.2%          | 221.3%         | 0.0100          | 0.64   | 51.9%                   | 0.43         |
| ElasticNet    | EW          | 33.1%  | 21.4%      |     1.44 | -15.7%         | 82.8%          | 157.3%         | 0.0038          | 0.20   | 59.3%                   | 1.06         |
| ElasticNet    | MV          | 40.6%  | 22.7%      |     1.62 | -19.4%         | 105.3%         | 211.3%         | 0.0038          | 0.20   | 63.0%                   | 1.29         |
| Random Forest | EW          | 16.2%  | 19.4%      |     0.87 | -19.3%         | 37.4%          | 223.6%         | 0.0208          | 0.98   | 55.6%                   | 0.27         |
| Random Forest | MV          | 21.2%  | 20.7%      |     1.03 | -21.8%         | 50.0%          | 263.6%         | 0.0208          | 0.98   | 59.3%                   | 0.40         |
| XGBoost       | EW          | 28.9%  | 19.1%      |     1.42 | -19.4%         | 70.9%          | 221.1%         | 0.0183          | 0.83   | 63.0%                   | 0.85         |
| XGBoost       | MV          | 31.4%  | 20.4%      |     1.44 | -19.7%         | 78.1%          | 251.5%         | 0.0183          | 0.83   | 55.6%                   | 0.78         |
| RNN           | EW          | 23.7%  | 19.9%      |     1.17 | -15.7%         | 56.8%          | 181.9%         | 0.0092          | 0.53   | 51.9%                   | 0.60         |
| RNN           | MV          | 25.6%  | 21.0%      |     1.19 | -17.1%         | 61.9%          | 228.1%         | 0.0092          | 0.53   | 51.9%                   | 0.51         |
| LSTM          | EW          | 19.0%  | 20.5%      |     0.95 | -20.7%         | 44.2%          | 158.3%         | 0.0137          | 0.74   | 51.9%                   | 0.36         |
| LSTM          | MV          | 27.5%  | 21.6%      |     1.24 | -15.8%         | 67.1%          | 213.5%         | 0.0137          | 0.74   | 55.6%                   | 0.61         |
| ARIMA         | EW          | 10.2%  | 22.7%      |     0.54 | -23.0%         | 22.7%          | 141.8%         | 0.0072          | 0.25   | 51.9%                   | -0.08        |
| ARIMA         | MV          | 11.4%  | 23.8%      |     0.57 | -24.1%         | 25.6%          | 202.0%         | 0.0072          | 0.25   | 63.0%                   | -0.10        |
| Benchmark     | -           | 7.6%   | 17.5%      |     0.51 | -20.1%         | 16.7%          |                |                 |        |                         |              |

Markowitz fell back to minimum variance (no positive mu among the picks) in 24 of 216 portfolio-periods.