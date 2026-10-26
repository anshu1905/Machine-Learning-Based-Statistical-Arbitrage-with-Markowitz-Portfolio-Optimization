# All variants: historical vs ML average vs benchmark

| variant                      | mu source   | weighting   | CAGR   | vol   |   Sharpe | max DD   | Sharpe, mean of 20 start days   |
|:-----------------------------|:------------|:------------|:-------|:------|---------:|:---------|:--------------------------------|
| Long-only, 252-day lookback  | Historical  | MV          | 9.5%   | 27.9% |     0.46 | -29.5%   | 0.16                            |
| Long-only, 252-day lookback  | Historical  | EW          | 9.0%   | 28.4% |     0.45 | -31.5%   | 0.16                            |
| Long-only, 252-day lookback  | ML average  | MV          | 22.8%  | 21.4% |     1.07 | -22.7%   | 0.81                            |
| Long-only, 252-day lookback  | ML average  | EW          | 25.1%  | 23.2% |     1.08 | -23.5%   | 0.86                            |
| Long-only, 252-day lookback  | Benchmark   | -           | 7.6%   | 17.5% |     0.51 | -20.1%   |                                 |
| Long-only, 20-day lookback   | Historical  | MV          | 23.9%  | 24.3% |     1.01 | -22.9%   | 0.75                            |
| Long-only, 20-day lookback   | Historical  | EW          | 23.0%  | 24.5% |     0.97 | -24.6%   | 0.71                            |
| Long-only, 20-day lookback   | ML average  | MV          | 29.5%  | 22.3% |     1.27 | -22.4%   | 0.92                            |
| Long-only, 20-day lookback   | ML average  | EW          | 25.1%  | 23.2% |     1.08 | -23.5%   | 0.86                            |
| Long-only, 20-day lookback   | Benchmark   | -           | 7.6%   | 17.5% |     0.51 | -20.1%   |                                 |
| Long-short, 252-day lookback | Historical  | MV          | 6.1%   | 23.8% |     0.37 | -33.4%   | -0.01                           |
| Long-short, 252-day lookback | Historical  | EW          | 5.5%   | 23.6% |     0.34 | -28.3%   | -0.08                           |
| Long-short, 252-day lookback | ML average  | MV          | 23.9%  | 15.8% |     1.43 | -15.5%   | 1.09                            |
| Long-short, 252-day lookback | ML average  | EW          | 21.9%  | 16.5% |     1.29 | -16.1%   | 0.97                            |
| Long-short, 252-day lookback | Benchmark   | -           | 7.6%   | 17.5% |     0.51 | -20.1%   |                                 |
| Long-short, 20-day lookback  | Historical  | MV          | 10.8%  | 23.4% |     0.56 | -24.6%   | 0.29                            |
| Long-short, 20-day lookback  | Historical  | EW          | 9.0%   | 23.2% |     0.49 | -20.2%   | 0.30                            |
| Long-short, 20-day lookback  | ML average  | MV          | 26.6%  | 17.0% |     1.47 | -17.8%   | 1.01                            |
| Long-short, 20-day lookback  | ML average  | EW          | 21.9%  | 16.5% |     1.29 | -16.1%   | 0.97                            |
| Long-short, 20-day lookback  | Benchmark   | -           | 7.6%   | 17.5% |     0.51 | -20.1%   |                                 |