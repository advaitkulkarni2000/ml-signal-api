# ML Signal API

A production-grade ML system that serves quantitative trading signals via a real-time REST API.

Built on a signal research framework with walk-forward validation, LightGBM alpha generation, isotonic calibration, Prometheus monitoring, and Docker containerisation.

![CI](https://github.com/advaitkulkarni2000/ml-signal-api/actions/workflows/ci.yml/badge.svg)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client                               │
│           POST /predict  {"tickers": ["AAPL", ...]}         │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    FastAPI Application                       │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  schemas.py  │   │  signals.py  │   │   model.py     │  │
│  │  Pydantic    │   │  6 alpha     │   │  LightGBM      │  │
│  │  validation  │   │  signals     │   │  + calibrator  │  │
│  └──────────────┘   └──────┬───────┘   └───────┬────────┘  │
│                            │                   │           │
│                     ┌──────▼───────────────────▼────────┐  │
│                     │         monitoring.py              │  │
│                     │  Prometheus counters, histograms   │  │
│                     └──────────────────────┬────────────┘  │
└────────────────────────────────────────────┼───────────────┘
                                             │ /metrics
┌────────────────────────────────────────────▼───────────────┐
│                      Prometheus                             │
│              Scrapes /metrics every 15s                     │
└────────────────────────────────────────────┬───────────────┘
                                             │ PromQL queries
┌────────────────────────────────────────────▼───────────────┐
│                       Grafana                               │
│        Latency, error rate, prediction drift dashboards     │
└─────────────────────────────────────────────────────────────┘
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Signal scores + predicted returns for a list of tickers |
| `/simulate` | POST | Long/short portfolio construction from signal rankings |
| `/health` | GET | Model status, uptime, prediction count |
| `/metrics` | GET | Prometheus metrics (latency, error rate, prediction drift) |
| `/docs` | GET | Interactive Swagger API documentation |

## Quick Start

### 1. Train the model

```bash
pip install -r requirements.txt
python -m training.train
```

MLflow tracks the run. View the experiment browser:
```bash
mlflow ui --port 5000
```
Open `http://localhost:5000` to compare runs, metrics, and artefacts.

### 2. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

### 3. Run the full stack (API + Prometheus + Grafana)

```bash
docker compose -f docker/docker-compose.yml up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

### 4. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "lookback_days": 252,
    "top_n": 3
  }'
```

**Example response:**
```json
{
  "signals": [
    {
      "ticker": "NVDA",
      "momentum_score": 1.82,
      "mean_reversion_score": -0.43,
      "volatility_score": -1.12,
      "composite_score": 0.021,
      "predicted_return": 0.021,
      "confidence": 0.68,
      "rank": 1
    }
  ],
  "model_version": "20240715_143022",
  "model_trained_at": "2024-07-15T14:30:22",
  "data_as_of": "2024-07-14",
  "computation_time_ms": 312.4,
  "tickers_requested": 7,
  "tickers_returned": 3
}
```

## Signals

Six cross-sectionally z-scored alpha signals (all features have mean 0, std 1 across the requested universe at inference time):

| Signal | Window | Description |
|--------|--------|-------------|
| `momentum_1m` | 21 days | Cross-sectional cumulative return momentum |
| `momentum_3m` | 63 days | Cross-sectional 3-month momentum |
| `momentum_6m` | 126 days | Cross-sectional 6-month momentum |
| `mean_reversion` | 5 days | Short-term reversal (negated 5-day return) |
| `volatility` | 21 days | Low-volatility signal (negated 21-day vol) |
| `vol_momentum` | 10 days | Volume-adjusted momentum proxy |

The LightGBM model is trained with walk-forward cross-validation (2-year expanding train / 3-month OOS) to prevent look-ahead bias. The isotonic calibrator maps raw model scores to calibrated probabilities.

## Monitoring

The `/metrics` endpoint exposes:
- `prediction_requests_total{status}` — request count by status (success/error)
- `request_latency_ms{endpoint}` — latency histogram with p50/p95/p99 quantiles
- `predicted_return_distribution` — distribution of model outputs (drift detection)
- `confidence_distribution` — distribution of calibrated confidence scores
- `data_fetch_errors_total{ticker}` — per-ticker data fetch failure count

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
ml-signal-api/
├── app/
│   ├── main.py         FastAPI routes, middleware, lifespan
│   ├── model.py        LightGBM model manager (singleton)
│   ├── signals.py      Signal computation + yfinance data fetching
│   ├── schemas.py      Pydantic request/response validation
│   └── monitoring.py   Prometheus metrics instrumentation
├── training/
│   └── train.py        Walk-forward training + MLflow logging
├── tests/
│   └── test_api.py     pytest unit + integration tests
├── docker/
│   ├── Dockerfile      Multi-stage container build
│   ├── docker-compose.yml  Full stack (API + Prometheus + Grafana)
│   └── prometheus.yml  Prometheus scrape config
├── .github/workflows/
│   └── ci.yml          GitHub Actions CI (test + docker build)
└── requirements.txt
```
