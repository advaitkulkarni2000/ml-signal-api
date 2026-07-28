"""
monitoring.py — Prometheus metrics instrumentation.

Prometheus is a pull-based monitoring system. It works like this:
1. Your app exposes a /metrics endpoint with current metric values
2. Prometheus server scrapes /metrics every N seconds
3. Grafana queries Prometheus and draws dashboards

--- WHY MONITORING IN PRODUCTION? ---
Without monitoring, you're flying blind. You won't know:
- Whether your model's predictions are getting worse over time (drift)
- How fast your API is responding (latency percentiles)
- How many requests are failing (error rate)
- What the distribution of predicted returns looks like (if it shifts,
  your features or data pipeline may be broken)

The four "Golden Signals" (Google SRE book):
1. Latency    — how long requests take
2. Traffic    — how many requests per second
3. Errors     — what fraction of requests fail
4. Saturation — how full are your resources

We implement all four here.
"""

from prometheus_client import (
    Counter,       # monotonically increasing count (requests, errors)
    Histogram,     # distribution of values (latency, predicted returns)
    Gauge,         # current value that can go up/down (model version, uptime)
    Summary,       # similar to Histogram but with quantile computation
)
import time
import functools
import logging

logger = logging.getLogger(__name__)


# ── METRIC DEFINITIONS ────────────────────────────────────────────────────────
# These are module-level objects — created once, updated throughout app lifetime.

# Counter: total number of prediction requests
# Counters only go up. They're reset when the process restarts.
# In Prometheus queries: rate(prediction_requests_total[5m]) = req/sec
PREDICTION_REQUESTS = Counter(
    name="prediction_requests_total",
    documentation="Total number of /predict endpoint calls",
    labelnames=["status"],  # "success" or "error"
)

# Histogram: distribution of request latency in milliseconds
# Prometheus histograms track: count, sum, and bucket counts
# Buckets: [10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1000ms, ∞]
# You can then query p50, p95, p99 latency
REQUEST_LATENCY = Histogram(
    name="request_latency_ms",
    documentation="Request latency in milliseconds",
    labelnames=["endpoint"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

# Histogram: distribution of predicted returns
# If this distribution shifts (e.g. suddenly predicting all positive returns),
# something is wrong with the features or data pipeline
PREDICTED_RETURN_DIST = Histogram(
    name="predicted_return_distribution",
    documentation="Distribution of model predicted returns",
    buckets=[-0.05, -0.03, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.03, 0.05],
)

# Histogram: distribution of confidence scores
# Should be roughly uniform [0,1] for a well-calibrated model
CONFIDENCE_DIST = Histogram(
    name="confidence_distribution",
    documentation="Distribution of model confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Counter: data fetch errors (yfinance failures)
DATA_FETCH_ERRORS = Counter(
    name="data_fetch_errors_total",
    documentation="Total number of yfinance data fetch failures",
    labelnames=["ticker"],
)

# Gauge: number of tickers currently being served
# Gauges can go up or down — current state rather than cumulative count
TICKERS_IN_REQUEST = Gauge(
    name="tickers_in_last_request",
    documentation="Number of tickers in the most recent request",
)

# Gauge: model version (as a timestamp for easy tracking)
MODEL_VERSION_GAUGE = Gauge(
    name="model_version_timestamp",
    documentation="Unix timestamp when current model was trained",
)

# Counter: cache hits (if we add caching later)
CACHE_HITS = Counter(
    name="cache_hits_total",
    documentation="Number of times result was served from cache",
)


# ── INSTRUMENTATION HELPERS ───────────────────────────────────────────────────

def track_request(endpoint: str):
    """
    Decorator that automatically measures latency and tracks request counts.

    Usage:
        @track_request("predict")
        async def predict_endpoint(request):
            ...

    This is the production pattern: wrap routes with instrumentation
    decorators so the monitoring code doesn't clutter business logic.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                PREDICTION_REQUESTS.labels(status="success").inc()
                return result
            except Exception as e:
                PREDICTION_REQUESTS.labels(status="error").inc()
                raise
            finally:
                # Finally always runs, even if exception raised
                elapsed_ms = (time.perf_counter() - start) * 1000
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed_ms)
        return wrapper
    return decorator


def record_predictions(predicted_returns: list[float], confidences: list[float]):
    """
    Record the distribution of model outputs for drift detection.

    If predicted_returns shifts significantly (e.g. suddenly mostly
    positive), something may have changed in the data pipeline.
    Monitoring the distribution over time catches this.
    """
    for ret in predicted_returns:
        PREDICTED_RETURN_DIST.observe(ret)
    for conf in confidences:
        CONFIDENCE_DIST.observe(conf)


def record_data_fetch_error(ticker: str):
    """Record a yfinance data fetch failure."""
    DATA_FETCH_ERRORS.labels(ticker=ticker).inc()
    logger.warning(f"Data fetch error recorded for {ticker}")
