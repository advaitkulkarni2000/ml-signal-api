"""
schemas.py — Data contracts for the API.

These classes define exactly what shape of data the API accepts
and what shape it returns. If a request doesn't match the schema,
FastAPI automatically returns a 422 error with a clear explanation
of what was wrong — before your code even runs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date


# ── REQUEST SCHEMAS (what comes IN) ──────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Input for the /predict endpoint.

    The user sends a list of tickers and we return signal scores
    and predicted returns for each one.
    """
    tickers: list[str] = Field(
        ...,                          # ... means required (no default)
        min_length=1,
        max_length=50,
        description="List of stock tickers, e.g. ['AAPL', 'MSFT', 'GOOGL']"
    )
    lookback_days: int = Field(
        default=252,                  # 1 trading year
        ge=60,                        # must be >= 60 days
        le=756,                       # must be <= 3 trading years
        description="Number of trading days of history to use for signal computation"
    )
    top_n: Optional[int] = Field(
        default=None,
        ge=1,
        description="If set, return only the top N tickers by predicted return"
    )

    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        """Force tickers to uppercase and remove duplicates."""
        cleaned = list(dict.fromkeys(t.upper().strip() for t in v))
        return cleaned

    model_config = {
        "json_schema_extra": {
            "example": {
                "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                "lookback_days": 252,
                "top_n": 3
            }
        }
    }


class SimulateRequest(BaseModel):
    """
    Input for the /simulate endpoint.

    Runs a portfolio simulation using the signal scores to rank
    stocks and construct a long/short portfolio.
    """
    tickers: list[str] = Field(..., min_length=5, max_length=100)
    lookback_days: int = Field(default=252, ge=60, le=756)
    long_short_n: int = Field(
        default=5,
        ge=1,
        description="Number of stocks in each leg of the long/short portfolio"
    )

    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        return list(dict.fromkeys(t.upper().strip() for t in v))


# ── RESPONSE SCHEMAS (what goes OUT) ─────────────────────────────────────────

class SignalScore(BaseModel):
    """Signal scores for a single ticker."""
    ticker: str
    momentum_score: float = Field(description="Cross-sectional momentum z-score")
    mean_reversion_score: float = Field(description="Z-score mean reversion signal")
    volatility_score: float = Field(description="Volatility regime signal")
    composite_score: float = Field(description="Model-weighted composite signal")
    predicted_return: float = Field(description="LightGBM predicted 1-day forward return")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence (calibrated probability)"
    )
    rank: int = Field(description="Rank among all requested tickers (1=best)")


class PredictResponse(BaseModel):
    """Full response from /predict endpoint."""
    signals: list[SignalScore]
    model_version: str
    model_trained_at: str
    data_as_of: str
    computation_time_ms: float
    tickers_requested: int
    tickers_returned: int
    warning: Optional[str] = None   # e.g. "3 tickers had insufficient data"


class SimulateResponse(BaseModel):
    """Response from /simulate endpoint."""
    long_tickers: list[str]
    short_tickers: list[str]
    long_scores: list[float]
    short_scores: list[float]
    expected_portfolio_return: float
    model_version: str
    data_as_of: str


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str                        # "healthy" or "degraded"
    model_loaded: bool
    model_version: str
    model_trained_at: str
    uptime_seconds: float
    total_predictions_served: int
    avg_latency_ms: float
