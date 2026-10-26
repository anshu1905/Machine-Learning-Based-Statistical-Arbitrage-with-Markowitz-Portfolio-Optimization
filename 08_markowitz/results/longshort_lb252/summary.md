# Top-10 + Markowitz backtest (long top-10 / short bottom-10, dollar-neutral, 252-day lookback): ML vs historical expected returns

Test period 2024-07-01 -> 2026-08-21, 27 rebalances every 20 trading days, cost 10 bps per unit of one-way turnover, Markowitz cap 25%, Sharpe with rf = 0.0%. EW = equal weight, MV = Markowitz max-Sharpe. `daily test IC` = mean daily Spearman correlation between the expected-return input (historical 252-day mean, or the model's prediction) and the realised 20-day excess return, over every test day; `IC t` = its Newey-West t-stat. `t vs Hist.` = paired t-stat of the per-period return difference against the historical-mean portfolio with the same weighting.

| source        | weighting   | CAGR   | ann. vol   |   Sharpe | max drawdown   | total return   | avg turnover   | daily test IC   | IC t   | beats Hist. (periods)   | t vs Hist.   |
|:--------------|:------------|:-------|:-----------|---------:|:---------------|:---------------|:---------------|:----------------|:-------|:------------------------|:-------------|
| Historical    | EW          | 5.5%   | 23.6%      |     0.34 | -28.3%         | 11.9%          | 147.0%         | 0.0008          | 0.03   |                         |              |
| Historical    | MV          | 6.1%   | 23.8%      |     0.37 | -33.4%         | 13.4%          | 175.2%         | 0.0008          | 0.03   |                         |              |
| Ridge         | EW          | 17.2%  | 20.7%      |     0.87 | -16.8%         | 39.7%          | 164.4%         | 0.0100          | 0.64   | 55.6%                   | 0.64         |
| Ridge         | MV          | 22.4%  | 20.3%      |     1.1  | -16.5%         | 53.3%          | 206.8%         | 0.0100          | 0.64   | 51.9%                   | 0.89         |
| ElasticNet    | EW          | 33.1%  | 21.4%      |     1.44 | -15.7%         | 82.8%          | 157.3%         | 0.0038          | 0.20   | 59.3%                   | 1.25         |
| ElasticNet    | MV          | 39.6%  | 21.0%      |     1.7  | -12.0%         | 102.4%         | 197.9%         | 0.0038          | 0.20   | 55.6%                   | 1.53         |
| Random Forest | EW          | 16.2%  | 19.4%      |     0.87 | -19.3%         | 37.4%          | 223.6%         | 0.0208          | 0.98   | 51.9%                   | 0.65         |
| Random Forest | MV          | 18.7%  | 18.8%      |     1.01 | -17.6%         | 43.6%          | 249.6%         | 0.0208          | 0.98   | 51.9%                   | 0.77         |
| XGBoost       | EW          | 28.9%  | 19.1%      |     1.42 | -19.4%         | 70.9%          | 221.1%         | 0.0183          | 0.83   | 63.0%                   | 1.24         |
| XGBoost       | MV          | 23.2%  | 19.1%      |     1.19 | -22.8%         | 55.4%          | 245.7%         | 0.0183          | 0.83   | 59.3%                   | 1.00         |
| RNN           | EW          | 23.7%  | 19.9%      |     1.17 | -15.7%         | 56.8%          | 181.9%         | 0.0092          | 0.53   | 59.3%                   | 1.03         |
| RNN           | MV          | 25.1%  | 19.6%      |     1.24 | -13.4%         | 60.6%          | 214.3%         | 0.0092          | 0.53   | 63.0%                   | 1.03         |
| LSTM          | EW          | 19.0%  | 20.5%      |     0.95 | -20.7%         | 44.2%          | 158.3%         | 0.0137          | 0.74   | 55.6%                   | 0.65         |
| LSTM          | MV          | 19.0%  | 19.6%      |     0.98 | -16.5%         | 44.3%          | 192.9%         | 0.0137          | 0.74   | 48.1%                   | 0.63         |
| ARIMA         | EW          | 10.2%  | 22.7%      |     0.54 | -23.0%         | 22.7%          | 141.8%         | 0.0072          | 0.25   | 66.7%                   | 0.59         |
| ARIMA         | MV          | 13.6%  | 22.9%      |     0.67 | -27.6%         | 30.9%          | 181.0%         | 0.0072          | 0.25   | 66.7%                   | 0.67         |
| Benchmark     | -           | 7.6%   | 17.5%      |     0.51 | -20.1%         | 16.7%          |                |                 |        |                         |              |

Markowitz fell back to minimum variance (no positive mu among the picks) in 24 of 216 portfolio-periods.