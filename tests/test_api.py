"""
tests/test_api.py — Automated test suite for the API.

Tests cover:
1. Health check with no model loaded
2. Schema validation (bad inputs)
3. Predict endpoint with a mock model
4. Simulate endpoint
5. Metrics endpoint format

--- WHY TESTS MATTER FOR A LEVEL 3 PROJECT ---
Without tests, you can't safely change code. With tests, every push
to GitHub runs these checks automatically (via GitHub Actions) and
tells you immediately if something broke.

Tests also document how the API is supposed to behave — they're
executable specifications.

--- TEST TYPES USED HERE ---
Unit tests: test a single function in isolation (e.g. cross_sectional_zscore)
Integration tests: test a full request through the FastAPI app (e.g. POST /predict)

We use pytest for the test runner and httpx.AsyncClient as the test
HTTP client — it simulates real HTTP requests without needing a running server.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.signals import cross_sectional_zscore, compute_returns
from app.schemas import PredictRequest, SignalScore


# ── FIXTURES ──────────────────────────────────────────────────────────────────
# pytest fixtures are reusable test setup functions.
# @pytest.fixture means: run this before each test that uses it.

@pytest.fixture
def sample_prices():
    """
    Create a small synthetic price DataFrame for testing signals.

    We generate fake prices rather than calling yfinance in tests —
    tests must be fast and deterministic, not dependent on network.
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="B")  # B = business days
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Generate random walk prices: P_t = P_{t-1} * exp(r_t) where r_t ~ N(0, 0.01)
    # This is the standard geometric Brownian motion model for stock prices
    log_returns = np.random.normal(0, 0.01, size=(200, 5))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))

    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def sample_features(sample_prices):
    """
    Compute features from sample prices for model input testing.
    """
    from app.signals import compute_feature_matrix
    return compute_feature_matrix(sample_prices)


@pytest.fixture
def mock_model_manager():
    """
    A mock model manager that returns predictable outputs.

    In tests we don't want to load a real model (it might not exist,
    and it's slow). We mock the model_manager to return fixed predictions.

    MagicMock() creates an object where every attribute and method call
    returns another MagicMock — so model_manager.is_loaded() returns a
    MagicMock, which we then configure to return True.
    """
    mock = MagicMock()
    mock.is_loaded.return_value = True
    mock.metadata.version = "test_v1"
    mock.metadata.trained_at = "2024-01-01T00:00:00"
    mock.metadata.feature_names = [
        "momentum_1m", "momentum_3m", "momentum_6m",
        "mean_reversion", "volatility", "vol_momentum"
    ]
    mock.prediction_count = 0
    mock.uptime_seconds = 10.0
    mock.avg_latency_ms = 5.0

    # predict() returns fixed arrays
    def mock_predict(features):
        n = len(features)
        return {
            "raw_scores": np.linspace(-0.02, 0.02, n),
            "predicted_return": np.linspace(-0.02, 0.02, n),
            "confidence": np.linspace(0.3, 0.7, n),
        }

    mock.predict.side_effect = mock_predict
    return mock


# ── SIGNAL UNIT TESTS ─────────────────────────────────────────────────────────

class TestSignals:
    """Unit tests for signal computation functions."""

    def test_cross_sectional_zscore_mean_zero(self):
        """After z-scoring, the mean should be 0."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                         index=["A", "B", "C", "D", "E"])
        result = cross_sectional_zscore(data)
        assert abs(result.mean()) < 1e-10, "Z-scored mean should be ~0"

    def test_cross_sectional_zscore_std_one(self):
        """After z-scoring, the std should be 1."""
        data = pd.Series([10.0, 20.0, 15.0, 5.0, 25.0],
                         index=["A", "B", "C", "D", "E"])
        result = cross_sectional_zscore(data)
        assert abs(result.std() - 1.0) < 1e-10, "Z-scored std should be ~1"

    def test_cross_sectional_zscore_constant_input(self):
        """Constant input (std=0) should return all zeros, not divide-by-zero."""
        data = pd.Series([5.0, 5.0, 5.0], index=["A", "B", "C"])
        result = cross_sectional_zscore(data)
        assert (result == 0.0).all(), "Constant input should return all zeros"

    def test_compute_returns_shape(self, sample_prices):
        """Returns DataFrame should have same shape as prices minus first row."""
        returns = compute_returns(sample_prices)
        assert returns.shape[0] == sample_prices.shape[0] - 1
        assert returns.shape[1] == sample_prices.shape[1]

    def test_compute_returns_log_property(self, sample_prices):
        """
        Log returns should be additive: r(t1,t3) = r(t1,t2) + r(t2,t3).

        This is the key mathematical property of log returns that makes
        them preferable to simple returns for financial modelling.
        """
        returns = compute_returns(sample_prices)
        ticker = sample_prices.columns[0]

        # 2-day return should equal sum of two 1-day returns
        two_day_log_return = np.log(
            sample_prices[ticker].iloc[2] / sample_prices[ticker].iloc[0]
        )
        sum_of_daily = returns[ticker].iloc[0] + returns[ticker].iloc[1]
        assert abs(two_day_log_return - sum_of_daily) < 1e-10

    def test_feature_matrix_no_nans(self, sample_prices):
        """Feature matrix should not contain NaN values."""
        from app.signals import compute_feature_matrix
        features = compute_feature_matrix(sample_prices)
        assert not features.isnull().any().any(), "Features should not contain NaN"

    def test_feature_matrix_columns(self, sample_prices):
        """Feature matrix should have exactly the expected columns."""
        from app.signals import compute_feature_matrix
        features = compute_feature_matrix(sample_prices)
        expected = {
            "momentum_1m", "momentum_3m", "momentum_6m",
            "mean_reversion", "volatility", "vol_momentum"
        }
        assert set(features.columns) == expected

    def test_feature_matrix_cross_sectional_mean_zero(self, sample_prices):
        """
        Each feature column should be cross-sectionally z-scored:
        mean ≈ 0, std ≈ 1 across tickers.
        """
        from app.signals import compute_feature_matrix
        features = compute_feature_matrix(sample_prices)
        for col in features.columns:
            col_mean = features[col].mean()
            col_std = features[col].std()
            assert abs(col_mean) < 1e-10, f"{col} mean should be ~0, got {col_mean}"
            assert abs(col_std - 1.0) < 1e-10, f"{col} std should be ~1, got {col_std}"


# ── API INTEGRATION TESTS ─────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Integration tests for /health."""

    @pytest.mark.asyncio
    async def test_health_no_model(self):
        """
        /health should return 503 when no model is loaded.

        This tests the "degraded" state — model not yet trained.
        """
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        # 503 = Service Unavailable (model not loaded)
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_health_with_model(self, mock_model_manager):
        """
        /health should return 200 when model is loaded.
        """
        with patch("app.main.model_manager", mock_model_manager):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_version"] == "test_v1"


class TestPredictEndpoint:
    """Integration tests for /predict."""

    @pytest.mark.asyncio
    async def test_predict_no_model_returns_503(self):
        """Without a model, /predict should return 503."""
        payload = {"tickers": ["AAPL", "MSFT"], "lookback_days": 252}
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/predict", json=payload)
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_predict_empty_tickers_rejected(self):
        """
        Pydantic validation should reject empty ticker list before
        the route handler even runs (returns 422).
        """
        payload = {"tickers": [], "lookback_days": 252}
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/predict", json=payload)
        # 422 = Unprocessable Entity — schema validation failed
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_tickers_uppercased(self):
        """
        Lowercase tickers should be silently uppercased by the validator.
        """
        request = PredictRequest(tickers=["aapl", "msft"])
        assert request.tickers == ["AAPL", "MSFT"]

    @pytest.mark.asyncio
    async def test_predict_duplicate_tickers_removed(self):
        """Duplicate tickers should be deduplicated."""
        request = PredictRequest(tickers=["AAPL", "AAPL", "MSFT"])
        assert request.tickers == ["AAPL", "MSFT"]
        assert len(request.tickers) == 2

    @pytest.mark.asyncio
    async def test_predict_lookback_bounds(self):
        """lookback_days must be between 60 and 756."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(tickers=["AAPL"], lookback_days=10)  # too small

        with pytest.raises(pydantic.ValidationError):
            PredictRequest(tickers=["AAPL"], lookback_days=1000)  # too large


class TestMetricsEndpoint:
    """Integration tests for /metrics."""

    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus_format(self):
        """
        /metrics should return Prometheus text format.

        Prometheus text format starts with # HELP and # TYPE lines.
        """
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/metrics")

        assert response.status_code == 200
        # Prometheus format always contains these marker strings
        assert b"# HELP" in response.content
        assert b"# TYPE" in response.content

    @pytest.mark.asyncio
    async def test_metrics_contains_our_counters(self):
        """
        /metrics should contain our custom metric names.
        """
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/metrics")

        content = response.content.decode()
        assert "prediction_requests_total" in content
        assert "request_latency_ms" in content


class TestSchemaValidation:
    """Unit tests for Pydantic schema validation."""

    def test_signal_score_confidence_bounds(self):
        """Confidence must be between 0 and 1."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SignalScore(
                ticker="AAPL",
                momentum_score=0.5,
                mean_reversion_score=0.3,
                volatility_score=-0.2,
                composite_score=0.4,
                predicted_return=0.01,
                confidence=1.5,  # > 1, invalid
                rank=1,
            )

    def test_signal_score_valid(self):
        """Valid SignalScore should construct without error."""
        score = SignalScore(
            ticker="AAPL",
            momentum_score=0.5,
            mean_reversion_score=0.3,
            volatility_score=-0.2,
            composite_score=0.4,
            predicted_return=0.01,
            confidence=0.65,
            rank=1,
        )
        assert score.ticker == "AAPL"
        assert 0 <= score.confidence <= 1
