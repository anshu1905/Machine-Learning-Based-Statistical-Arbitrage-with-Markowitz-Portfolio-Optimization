"""Project-wide paths, split dates and column groups.

Everything that more than one model folder needs to agree on lives here, so that
Ridge, ElasticNet, RF, XGBoost, RNN, LSTM and ARIMA are all evaluated on the
same rows, the same target and the same train / validation / test periods.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "features_by_stock.xlsx"
DATA_DIR = ROOT / "data"
PANEL_RAW = DATA_DIR / "panel_raw.parquet"        # xlsx -> parquet, untouched
PANEL_CLEAN = DATA_DIR / "panel_clean.parquet"    # + extra features, flags, splits
FEATURE_IC = DATA_DIR / "feature_ic_train.csv"    # univariate IC table (train only)
FEATURE_CORR = DATA_DIR / "feature_corr_train.csv"  # cross-sectional rank corr (train only)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
HORIZON = 20                                  # trading days
TARGET = "target_excess_ret_20d"              # fwd_ret_20d - bench_fwd_ret
TARGET_CLASS = "target_outperform"            # 1{TARGET > 0}

# ---------------------------------------------------------------------------
# Splits (all feature selection is done on TRAIN only)
#   2018 is burn-in: mom_252d / dist_ma200 need ~1 year of history.
#   A purge gap of HORIZON trading days is removed before VAL and TEST so that
#   no 20-day target window straddles two periods.
# ---------------------------------------------------------------------------
BURN_IN_END = "2018-12-31"
TRAIN_START, TRAIN_END = "2019-01-01", "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2024-06-30"
TEST_START = "2024-07-01"

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
ID_COLS = ["Date", "Symbol"]
RAW_OHLCV = ["Open", "High", "Low", "Close", "Volume", "VWAP"]

# Anything computed with information after t. Never a feature.
LEAKAGE = ["fwd_ret_20d", "bench_fwd_ret", "target_excess_ret_20d", "target_outperform"]

# Exact duplicates found in the audit (column -> the column it duplicates).
DUPLICATES = {"candle_body": "intraday_return"}

# Benchmark features: identical for every stock on a given date. They carry
# regime information but cannot rank stocks on their own.
MARKET = [
    "r_b", "bench_ret_20d", "bench_mom_60d", "bench_mom_120d", "bench_vol_20d",
    "bench_vol_60d", "bench_dist_ma200", "bench_drawdown_252d",
]

# Pre-computed cross-sectional ranks shipped in the file. They are rank
# transforms of columns already in STOCK_BASE, so they are redundant once we
# apply our own cross-sectional transform.
PRE_RANKED = [
    "rank_mom_20d", "rank_mom_60d", "rank_rel_mom_60d", "rank_vol_20d",
    "rank_resid_ret_60d", "rank_log_traded_value_med_20d",
]

# Stock-level features shipped in features_by_stock.xlsx.
STOCK_BASE = [
    "ret_1d", "ret_5d", "mom_20d", "mom_60d", "mom_120d", "mom_252d",
    "rel_mom_20d", "rel_mom_60d", "rel_mom_120d",
    "vol_20d", "vol_60d", "downside_vol_20d", "vol_ratio_20_60", "parkinson_vol_20d",
    "vol_adj_mom_60d", "mom_accel_20d",
    "dist_high_20d", "dist_high_60d", "dist_low_60d",
    "dist_ma20", "dist_ma50", "dist_ma200", "ma50_ma200_ratio", "ma50_slope_20d",
    "trend_consistency_60d",
    "log_volume", "log_traded_value_med_20d", "rel_volume_20d", "volume_trend_20_60",
    "price_volume_interaction",
    "intraday_return", "intraday_range", "overnight_gap_mean_20d", "gap_vol_20d",
    "clv_mean_20d",
    "beta_60d", "beta_120d", "corr_60d", "rel_vol_60d", "resid_ret_60d",
]

# Extra stock-level features from statarb_features_ranked.pdf that were not in
# the file (computed in common/features_extra.py, causal: data up to t only).
STOCK_EXTRA = [
    "amihud_log_20d",      # Tier 2  Amihud illiquidity (log)
    "yz_vol_20d",          # Tier 2  Yang-Zhang volatility
    "variance_ratio_5_60", # Tier 2  variance ratio VR(5) on 60d window
    "ret_autocorr_60d",    # Tier 3  lag-1 return autocorrelation
    "overnight_ret",       # Tier 3  ln(O_t / C_{t-1})
    "vol_ret_corr_20d",    # Tier 3  corr(|r|, V)
    "rsi_14",              # Tier 2  Wilder RSI
    "adx_14",              # Tier 3  ADX trend strength (regime filter)
    "mfi_14",              # Tier 3  money flow index
    "cmf_20",              # Tier 3  Chaikin money flow
]

STOCK_CANDIDATES = STOCK_BASE + STOCK_EXTRA

# Tier from the ranked-features PDF (1 = critical ... 4 = low), used only for
# reporting; selection is data driven.
PDF_TIER = {
    "ret_1d": 2, "ret_5d": 3, "mom_20d": 3, "mom_60d": 3, "mom_120d": 3, "mom_252d": 3,
    "rel_mom_20d": 2, "rel_mom_60d": 2, "rel_mom_120d": 2,
    "vol_20d": 2, "vol_60d": 2, "downside_vol_20d": 2, "vol_ratio_20_60": 2,
    "parkinson_vol_20d": 3, "vol_adj_mom_60d": 3, "mom_accel_20d": 2,
    "dist_high_20d": 4, "dist_high_60d": 4, "dist_low_60d": 4,
    "dist_ma20": 2, "dist_ma50": 2, "dist_ma200": 2, "ma50_ma200_ratio": 4,
    "ma50_slope_20d": 4, "trend_consistency_60d": 4,
    "log_volume": 4, "log_traded_value_med_20d": 4, "rel_volume_20d": 2,
    "volume_trend_20_60": 4, "price_volume_interaction": 3,
    "intraday_return": 3, "intraday_range": 4, "overnight_gap_mean_20d": 3,
    "gap_vol_20d": 3, "clv_mean_20d": 3,
    "beta_60d": 2, "beta_120d": 2, "corr_60d": 2, "rel_vol_60d": 2, "resid_ret_60d": 1,
    "amihud_log_20d": 2, "yz_vol_20d": 2, "variance_ratio_5_60": 2,
    "ret_autocorr_60d": 3, "overnight_ret": 3, "vol_ret_corr_20d": 3,
    "rsi_14": 2, "adx_14": 3, "mfi_14": 3, "cmf_20": 3,
}

SEED = 42
