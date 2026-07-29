"""
signals.py — Signal computation engine.

Uses a browser-spoofing requests session to avoid Yahoo Finance rate limiting.
Downloads tickers one at a time using yf.Ticker(session=session) which is
more reliable than yf.download() for avoiding 429/403 errors.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ── BROWSER SESSION ───────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    """
    Create a requests session with browser-like headers.

    Yahoo Finance blocks requests that look like bots (no User-Agent).
    Adding a real browser User-Agent string bypasses this check.
    The session object is reused across all ticker downloads so we
    don't create a new TCP connection for every request.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    return session


# ── DATA FETCHING ─────────────────────────────────────────────────────────────

def fetch_price_data(
    tickers: list[str],
    lookback_days: int = 252,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Download adjusted close prices for all tickers using browser session.

    Downloads one ticker at a time with a short delay between each.
    This is slower than bulk download but avoids Yahoo Finance rate limits.

    Returns:
        prices: DataFrame (rows=dates, cols=tickers)
        failed: list of tickers that could not be fetched
    """
    # Map lookback days to yfinance period string
    period_map = {60: "3mo", 126: "6mo", 252: "1y", 504: "2y", 756: "3y"}
    period = "1y"
    for days, p in sorted(period_map.items()):
        if lookback_days <= days:
            period = p
            break

    logger.info(f"Fetching {len(tickers)} tickers one by one, period={period}")

    session = make_session()
    yf.set_tz_cache_location("/tmp")

    all_prices = {}
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            # Use Ticker object with session — more reliable than yf.download
            t = yf.Ticker(ticker, session=session)
            hist = t.history(period=period, auto_adjust=True)

            if hist.empty or len(hist) < 60:
                logger.warning(f"{ticker}: insufficient data ({len(hist)} rows)")
                failed.append(ticker)
                continue

            # history() returns a DataFrame with columns: Open High Low Close Volume
            all_prices[ticker] = hist["Close"]
            logger.info(f"  [{i+1}/{len(tickers)}] {ticker}: {len(hist)} rows OK")

        except Exception as e:
            logger.warning(f"  [{i+1}/{len(tickers)}] {ticker}: failed — {e}")
            failed.append(ticker)

        # Short delay between requests to avoid rate limiting
        # 0.5 seconds gives ~2 requests/sec which Yahoo tolerates
        time.sleep(0.5)

    if not all_prices:
        raise ValueError(
            "No price data fetched for any ticker. "
            "Check network connection or try again in a few minutes."
        )

    prices = pd.DataFrame(all_prices)
    prices.index = pd.to_datetime(prices.index)

    # Remove timezone info from index (causes issues with some pandas operations)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    prices = prices.dropna(how="all")

    logger.info(
        f"Fetch complete: {len(all_prices)} succeeded, "
        f"{len(failed)} failed"
        + (f" ({failed})" if failed else "")
    )

    return prices, failed


# ── RETURN COMPUTATION ────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns: log(P_t / P_{t-1})

    Additive across time: weekly return = sum of daily log returns.
    Approximately normally distributed for small price changes.
    """
    return np.log(prices / prices.shift(1)).dropna(how="all")


# ── CROSS-SECTIONAL Z-SCORING ─────────────────────────────────────────────────

def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """
    Normalise across tickers at a single point in time.

    result_i = (x_i - mean(x)) / std(x)

    Cross-sectional (not time-series) to avoid look-ahead bias:
    mean and std are computed across tickers TODAY, not across history.
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


# ── SIGNAL FUNCTIONS ──────────────────────────────────────────────────────────

def momentum_signal(returns: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Cross-sectional momentum: cumulative return over window trading days.
    window=21 ≈ 1 month, 63 ≈ 3 months, 126 ≈ 6 months.
    """
    return cross_sectional_zscore(returns.tail(window).sum())


def mean_reversion_signal(returns: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Short-term mean reversion: negated 5-day return.
    Highest-IC signal in the original framework (ICIR=1.81 at 1-day horizon).
    """
    return cross_sectional_zscore(-returns.tail(window).sum())


def volatility_signal(returns: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Low-volatility anomaly: lower-vol stocks tend to outperform.
    Negated so higher signal = lower volatility = stronger long candidate.
    """
    return cross_sectional_zscore(-returns.tail(window).std())


def volume_momentum_signal(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 10,
) -> pd.Series:
    """
    Momentum adjusted by inverse volatility.
    High-vol price moves are noisier signals than low-vol moves.
    """
    mom = returns.tail(window).sum()
    vol_proxy = returns.tail(window).std().replace(0, np.nan)
    return cross_sectional_zscore((mom / vol_proxy).fillna(0))


# ── FEATURE MATRIX ────────────────────────────────────────────────────────────

def compute_feature_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 6-feature matrix for LightGBM inference.

    Each row = one ticker.
    Each column = one cross-sectionally z-scored signal.
    All features have mean≈0, std≈1 across the requested ticker universe.
    """
    returns = compute_returns(prices)

    features = pd.DataFrame(index=prices.columns)
    features["momentum_1m"]    = momentum_signal(returns, window=21)
    features["momentum_3m"]    = momentum_signal(returns, window=63)
    features["momentum_6m"]    = momentum_signal(returns, window=126)
    features["mean_reversion"] = mean_reversion_signal(returns, window=5)
    features["volatility"]     = volatility_signal(returns, window=21)
    features["vol_momentum"]   = volume_momentum_signal(prices, returns, window=10)

    features = features.dropna()
    logger.info(f"Feature matrix: {features.shape[0]} tickers x {features.shape[1]} signals")
    return features