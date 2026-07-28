"""
main.py — FastAPI application: routes, lifespan, dependency injection.

This is the entry point of the web server. It wires together:
- The model manager (model.py)
- The signal engine (signals.py)
- The monitoring layer (monitoring.py)
- The request/response schemas (schemas.py)

Into a set of HTTP endpoints that clients can call.

--- HOW FASTAPI WORKS ---
FastAPI is built on Starlette (ASGI framework) and Pydantic.
You define routes with decorators (@app.get, @app.post).
FastAPI automatically:
  1. Parses incoming JSON into your Pydantic schema
  2. Validates it (returns 422 if invalid, before your code runs)
  3. Calls your route function
  4. Serialises the return value to JSON
  5. Generates OpenAPI docs at /docs automatically

The whole thing runs on Uvicorn, an ASGI server that handles
async Python code efficiently using an event loop.
"""

import time
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.schemas import (
    PredictRequest, PredictResponse, SignalScore,
    SimulateRequest, SimulateResponse,
    HealthResponse,
)
from app.model import model_manager
from app.signals import fetch_price_data, compute_feature_matrix
from app.monitoring import (
    track_request, record_predictions, record_data_fetch_error,
    TICKERS_IN_REQUEST,
)

# Configure logging: format, level, output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── LIFESPAN: startup and shutdown logic ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs BEFORE the first request (startup) and AFTER the
    last request (shutdown).

    --- WHY LIFESPAN INSTEAD OF @app.on_event? ---
    @app.on_event("startup") is deprecated. The modern way is the
    lifespan context manager. It uses Python's async context manager
    protocol (async with / yield) to clearly separate startup from
    shutdown code.

    Everything BEFORE yield = startup.
    Everything AFTER yield = shutdown.
    """
    # ── STARTUP ──────────────────────────────────────────────────────────
    logger.info("Starting ml-signal-api...")

    # Load the model into memory (happens once, before any requests)
    loaded = model_manager.load()
    if loaded:
        logger.info(f"Model ready: version={model_manager.metadata.version}")
    else:
        logger.warning(
            "No trained model found. /predict will return 503 until "
            "training/train.py is run."
        )

    logger.info("API startup complete. Serving requests.")

    yield  # <-- server is running, handling requests

    # ── SHUTDOWN ─────────────────────────────────────────────────────────
    logger.info("Shutting down ml-signal-api...")
    # Clean up resources (database connections, file handles, etc.)
    # In our case nothing to clean up, but the pattern is important


# ── APP INITIALISATION ────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Signal API",
    description=(
        "Production ML system serving quantitative trading signals. "
        "Built on a signal research framework with walk-forward validation "
        "and LightGBM alpha generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # interactive Swagger UI at /docs
    redoc_url="/redoc",    # alternative ReDoc UI at /redoc
)


# ── MIDDLEWARE: request logging ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware runs on EVERY request, before and after the route handler.

    This is where you add cross-cutting concerns:
    - Request logging (who called what when)
    - Authentication checking
    - Rate limiting
    - Request ID injection (for tracing)

    call_next(request) passes the request to the actual route handler.
    """
    start = time.perf_counter()
    request_id = str(int(time.time() * 1000))  # simple request ID

    response = await call_next(request)

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        f"req_id={request_id} | "
        f"method={request.method} | "
        f"path={request.url.path} | "
        f"status={response.status_code} | "
        f"latency={elapsed:.1f}ms"
    )
    # Add latency to response headers (useful for debugging)
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    return response


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
async def health_check():
    """
    Health check endpoint. Called by:
    - Kubernetes liveness probes (is the container alive?)
    - Load balancers (should traffic be routed here?)
    - Monitoring dashboards

    Returns 200 if healthy, 503 if model not loaded.
    """
    is_healthy = model_manager.is_loaded()

    response = HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model_loaded=is_healthy,
        model_version=model_manager.metadata.version,
        model_trained_at=model_manager.metadata.trained_at,
        uptime_seconds=model_manager.uptime_seconds,
        total_predictions_served=model_manager.prediction_count,
        avg_latency_ms=model_manager.avg_latency_ms,
    )

    if not is_healthy:
        # 503 Service Unavailable — load balancer will stop sending traffic
        raise HTTPException(status_code=503, detail=response.dict())

    return response


# ── METRICS ENDPOINT ──────────────────────────────────────────────────────────

@app.get("/metrics", tags=["Infrastructure"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Prometheus scrapes this URL every 15 seconds.
    Returns all registered metrics in Prometheus text format.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── PREDICT ENDPOINT ──────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Signals"])
async def predict(request: PredictRequest):
    """
    Main prediction endpoint.

    Given a list of tickers, fetches price data, computes signals,
    runs LightGBM inference, and returns ranked signal scores with
    calibrated confidence for each ticker.

    ---
    **Example request:**
    ```json
    {
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "lookback_days": 252,
        "top_n": 3
    }
    ```
    """
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training/train.py first."
        )

    start = time.perf_counter()
    TICKERS_IN_REQUEST.set(len(request.tickers))

    # ── Step 1: Fetch price data ──────────────────────────────────────────
    try:
        prices, failed_tickers = fetch_price_data(
            request.tickers,
            request.lookback_days
        )
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {str(e)}")

    for t in failed_tickers:
        record_data_fetch_error(t)

    if prices.empty or len(prices.columns) == 0:
        raise HTTPException(
            status_code=422,
            detail="No valid price data could be fetched for any requested ticker."
        )

    # ── Step 2: Compute features ──────────────────────────────────────────
    try:
        features = compute_feature_matrix(prices)
    except Exception as e:
        logger.error(f"Feature computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {str(e)}")

    if features.empty:
        raise HTTPException(
            status_code=422,
            detail="Could not compute features. Tickers may have insufficient history."
        )

    # ── Step 3: Model inference ───────────────────────────────────────────
    try:
        predictions = model_manager.predict(features)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # ── Step 4: Build ranked response ─────────────────────────────────────
    tickers = features.index.tolist()
    raw_scores = predictions["predicted_return"]
    confidence = predictions["confidence"]

    # Rank: 1 = highest predicted return
    ranks = len(tickers) - np.argsort(np.argsort(raw_scores))

    signals = []
    for i, ticker in enumerate(tickers):
        signals.append(SignalScore(
            ticker=ticker,
            momentum_score=float(features.loc[ticker, "momentum_1m"]),
            mean_reversion_score=float(features.loc[ticker, "mean_reversion"]),
            volatility_score=float(features.loc[ticker, "volatility"]),
            composite_score=float(raw_scores[i]),
            predicted_return=float(raw_scores[i]),
            confidence=float(confidence[i]),
            rank=int(ranks[i]),
        ))

    # Sort by rank (best first)
    signals.sort(key=lambda x: x.rank)

    # Apply top_n filter if requested
    if request.top_n is not None:
        signals = signals[:request.top_n]

    # Record distributions for monitoring
    record_predictions(
        [s.predicted_return for s in signals],
        [s.confidence for s in signals]
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    warning = None
    if failed_tickers:
        warning = f"{len(failed_tickers)} tickers had insufficient data: {failed_tickers}"

    return PredictResponse(
        signals=signals,
        model_version=model_manager.metadata.version,
        model_trained_at=model_manager.metadata.trained_at,
        data_as_of=str(prices.index[-1].date()),
        computation_time_ms=round(elapsed_ms, 2),
        tickers_requested=len(request.tickers),
        tickers_returned=len(signals),
        warning=warning,
    )


# ── SIMULATE ENDPOINT ─────────────────────────────────────────────────────────

@app.post("/simulate", response_model=SimulateResponse, tags=["Signals"])
async def simulate(request: SimulateRequest):
    """
    Portfolio simulation endpoint.

    Constructs a long/short portfolio: long the top N tickers by
    predicted return, short the bottom N.

    This mirrors your backtesting framework's L/S Q5-Q1 portfolio.
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    prices, failed = fetch_price_data(request.tickers, request.lookback_days)
    features = compute_feature_matrix(prices)
    predictions = model_manager.predict(features)

    tickers = features.index.tolist()
    scores = predictions["predicted_return"]

    # Sort by score
    ranked = sorted(zip(tickers, scores), key=lambda x: x[1], reverse=True)

    n = min(request.long_short_n, len(ranked) // 2)
    long_leg  = ranked[:n]
    short_leg = ranked[-n:]

    return SimulateResponse(
        long_tickers=[t for t, _ in long_leg],
        short_tickers=[t for t, _ in short_leg],
        long_scores=[float(s) for _, s in long_leg],
        short_scores=[float(s) for _, s in short_leg],
        expected_portfolio_return=float(
            np.mean([s for _, s in long_leg]) - np.mean([s for _, s in short_leg])
        ),
        model_version=model_manager.metadata.version,
        data_as_of=str(prices.index[-1].date()),
    )


# ── ROOT ──────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Infrastructure"])
async def root():
    """API root — basic info."""
    return {
        "name": "ML Signal API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }
