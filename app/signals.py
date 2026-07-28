"""
signals.py — Signal computation engine.

This is the core ML/quantitative logic: takes raw price data and
computes alpha signals for each stock. These signals are then fed
into the LightGBM model for final return prediction.

Signals implemented (matching your original signal research framework):
  1. Cross-sectional momentum (1-month, 3-month, 6-month)
  2. Z-score mean reversion (short-term)
  3. Volatility regime (rolling vol)
  4. Volume-adjusted momentum

Each signal is cross-sectionally z-scored (mean 0, std 1 across all
tickers at each time step). This is critical: we normalise WITHIN
each date, not across time, to avoid look-ahead bias.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── DATA FETCHING ─────────────────────────────────────────────────────────────

def fetch_price_data(
    tickers: list[str],
    lookback_days: int = 252
) -> tuple[pd.DataFrame, list[str]]:
    """
    Download OHLCV data from Yahoo Finance for all tickers.

    Returns:
        prices: DataFrame with adjusted close prices (rows=dates, cols=tickers)
        failed: list of tickers that couldn't be fetched

    --- WHY ADJUSTED CLOSE? ---
    Raw close prices include corporate actions: stock splits, dividends.
    If AAPL splits 4:1, the raw price drops 75% overnight — not a real
    return. Adjusted close prices retroactively correct for these events
    so that returns computed from consecutive adjusted closes are real
    economic returns. Always use adjusted close for return computation.
    """
    # yfinance accepts a list of tickers and a period
    # period="1y" = 1 calendar year ≈ 252 trading days
    period_map = {
        60:  "3mo",
        126: "6mo",
        252: "1y",
        504: "2y",
        756: "3y",
    }
    # Find closest period
    period = "1y"
    for days, p in sorted(period_map.items()):
        if lookback_days <= days:
            period = p
            break

    logger.info(f"Fetching {len(tickers)} tickers, period={period}")

    try:
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,    # use adjusted close
            progress=False,      # suppress download progress bar
            threads=True,        # parallel downloads
        )
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        raise

    # yfinance returns MultiIndex columns when multiple tickers
    # Shape: (dates, tickers) for single price field
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers[:1]

    # Find tickers that returned insufficient data
    failed = []
    valid_tickers = []
    for t in tickers:
        if t not in prices.columns:
            failed.append(t)
        elif prices[t].dropna().shape[0] < 60:
            logger.warning(f"{t}: only {prices[t].dropna().shape[0]} rows, skipping")
            failed.append(t)
        else:
            valid_tickers.append(t)

    prices = prices[valid_tickers].dropna(how="all")

    logger.info(f"Fetched {len(valid_tickers)} tickers, {len(failed)} failed")
    return prices, failed


# ── RETURN COMPUTATION ────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from price series.

    Log return = log(P_t / P_{t-1})

    --- WHY LOG RETURNS? ---
    Simple returns r = (P_t - P_{t-1}) / P_{t-1} have two problems:
    1. Not additive across time: a 50% gain then a 50% loss ≠ 0%.
    2. Bounded below at -100% but unbounded above — asymmetric.

    Log returns solve both: log(P_t/P_{t-1}) ARE additive across time
    (log returns over a week = sum of daily log returns), and they're
    approximately normally distributed for small changes.

    For signal computation we use log returns throughout.
    """
    return np.log(prices / prices.shift(1)).dropna(how="all")


# ── CROSS-SECTIONAL Z-SCORING ─────────────────────────────────────────────────

def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """
    Z-score a series cross-sectionally: subtract mean, divide by std.

    result_i = (x_i - mean(x)) / std(x)

    --- WHY CROSS-SECTIONAL, NOT TIME-SERIES? ---
    If we z-score using historical mean/std (time-series), we'd be using
    future information at each point — because the mean computed over
    the full history includes data from after the prediction date.
    Cross-sectional z-scoring computes mean/std ACROSS all tickers
    at a SINGLE point in time. No future information leaks in.

    This is one of the most common bugs in quant research: normalising
    across time instead of across the cross-section.
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


# ── INDIVIDUAL SIGNALS ────────────────────────────────────────────────────────

def momentum_signal(returns: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Cross-sectional momentum: cumulative return over [window] days.

    For each ticker, compute the sum of log returns over the window.
    Then z-score across tickers at this point in time.

    Intuition: stocks that have gone up recently tend to keep going up
    (momentum effect). This is one of the most robust documented
    anomalies in academic finance.

    window=21 ≈ 1 month, window=63 ≈ 3 months, window=126 ≈ 6 months.
    """
    cumulative = returns.tail(window).sum()  # sum of log returns over window
    return cross_sectional_zscore(cumulative)


def mean_reversion_signal(returns: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Short-term mean reversion: negative of recent return.

    Stocks that have dropped sharply in the short term (1-2 weeks)
    tend to bounce back — the opposite of momentum at short horizons.

    In your original framework, this signal had ICIR=1.81 at 1-day
    horizon and decayed to 0.38 at 21 days. This is the highest-IC
    signal in the framework.
    """
    short_return = returns.tail(window).sum()
    return cross_sectional_zscore(-short_return)  # negate: high recent drop → positive signal


def volatility_signal(returns: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Volatility regime signal: low-vol stocks tend to outperform.

    This is the "low-volatility anomaly" — counter to CAPM theory,
    lower volatility stocks have historically delivered better
    risk-adjusted returns. We z-score and negate (lower vol → higher signal).
    """
    vol = returns.tail(window).std()
    return cross_sectional_zscore(-vol)  # negate: lower vol → higher signal


def volume_momentum_signal(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 10
) -> pd.Series:
    """
    Volume-adjusted momentum: price momentum weighted by trading volume.

    High-volume up moves are stronger signals than low-volume up moves.
    We can't directly get volume from our current setup without extra
    API calls, so we approximate using price volatility as a proxy.
    """
    # Proxy: momentum weighted by inverse volatility (high-vol moves less reliable)
    mom = returns.tail(window).sum()
    vol_proxy = returns.tail(window).std().replace(0, np.nan)
    adjusted = mom / vol_proxy
    return cross_sectional_zscore(adjusted.fillna(0))


# ── FEATURE MATRIX ────────────────────────────────────────────────────────────

def compute_feature_matrix(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full feature matrix for model inference.

    Each row = one ticker. Each column = one signal.
    This is the input to the LightGBM model.

    --- ML SIDE: FEATURE ENGINEERING ---
    Feature engineering is often more important than model choice.
    A simple model with good features beats a complex model with bad
    features almost every time. Here we compute:
    - 3 momentum signals at different horizons (1mo, 3mo, 6mo)
    - 1 mean reversion signal (5-day)
    - 1 volatility signal (21-day)
    - 1 volume-momentum signal (10-day)
    = 6 features total

    All features are cross-sectionally z-scored → mean 0, std 1.
    This means the LightGBM model sees relative rankings between
    stocks, not absolute levels. This is crucial because absolute
    price levels are meaningless for cross-sectional prediction.
    """
    returns = compute_returns(prices)

    features = pd.DataFrame(index=prices.columns)

    features["momentum_1m"]     = momentum_signal(returns, window=21)
    features["momentum_3m"]     = momentum_signal(returns, window=63)
    features["momentum_6m"]     = momentum_signal(returns, window=126)
    features["mean_reversion"]  = mean_reversion_signal(returns, window=5)
    features["volatility"]      = volatility_signal(returns, window=21)
    features["vol_momentum"]    = volume_momentum_signal(prices, returns, window=10)

    # Drop any tickers where we couldn't compute all features
    features = features.dropna()

    logger.info(f"Feature matrix shape: {features.shape}")
    return features
