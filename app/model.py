"""
model.py — Model management: loading, inference, calibration.

This module handles everything related to the LightGBM model:
- Loading a trained model from disk (or MLflow registry)
- Running inference on a feature matrix
- Calibrating raw model outputs to proper probabilities
- Tracking model metadata (version, training date)

--- THE SINGLETON PATTERN ---
We load the model ONCE at startup and keep it in memory.
Loading a model takes ~100-500ms. If we loaded it on every request,
a service handling 100 requests/second would waste 10-50 seconds
per second just loading models. The singleton ensures one load,
then reuse for every subsequent request.
"""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import logging
import time
from pathlib import Path
from typing import Optional
from sklearn.isotonic import IsotonicRegression
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Path where trained model artefacts live inside the container
import os
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_PATH = MODEL_DIR / "signal_model.pkl"
CALIBRATOR_PATH = MODEL_DIR / "calibrator.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


# ── MODEL METADATA ────────────────────────────────────────────────────────────

@dataclass
class ModelMetadata:
    """
    Tracks everything about a specific trained model version.

    In production you MUST know: which model is running, when it was
    trained, on what data, with what performance. Without this, when
    something goes wrong you can't diagnose whether it's a model issue
    or a data issue.
    """
    version: str = "unknown"
    trained_at: str = "unknown"
    training_tickers: int = 0
    training_days: int = 0
    val_sharpe: float = 0.0
    val_ic: float = 0.0
    feature_names: list = field(default_factory=list)
    mlflow_run_id: Optional[str] = None


# ── THE MODEL MANAGER (Singleton) ────────────────────────────────────────────

class SignalModelManager:
    """
    Manages the loaded model and its lifecycle.

    This is a singleton — only ONE instance exists in the entire
    application. FastAPI creates this once at startup via lifespan,
    and all requests share the same loaded model.

    --- WHY A CLASS INSTEAD OF GLOBAL VARIABLES? ---
    You could use module-level globals: model = None. But a class
    gives you:
    1. Encapsulation: model, calibrator, metadata all live together
    2. Methods: load(), predict(), reload() are logically grouped
    3. Thread safety: attributes are clearly owned by one object
    4. Testability: you can mock the class in tests
    """

    def __init__(self):
        self.model: Optional[lgb.Booster] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.metadata: ModelMetadata = ModelMetadata()
        self._loaded = False
        self._prediction_count = 0
        self._total_latency_ms = 0.0
        self._load_time = time.time()

    def load(self, model_path: Path = MODEL_PATH) -> bool:
        """
        Load model from disk into memory.

        Returns True if successful, False if model file doesn't exist
        (first run, before training).
        """
        if not model_path.exists():
            logger.warning(
                f"Model file not found at {model_path}. "
                "Run training/train.py first."
            )
            return False

        try:
            logger.info(f"Loading model from {model_path}")
            start = time.perf_counter()

            # joblib is the standard way to serialise sklearn/lgbm objects
            # It's faster and more memory-efficient than pickle for numpy arrays
            self.model = joblib.load(model_path)

            if CALIBRATOR_PATH.exists():
                self.calibrator = joblib.load(CALIBRATOR_PATH)
                logger.info("Calibrator loaded")

            if METADATA_PATH.exists():
                import json
                with open(METADATA_PATH) as f:
                    meta_dict = json.load(f)
                self.metadata = ModelMetadata(**meta_dict)

            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Model loaded in {elapsed:.1f}ms | version={self.metadata.version}")
            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Run inference and return predictions with confidence scores.

        Returns:
            dict with keys:
              'raw_scores'      - LightGBM raw leaf scores (unbounded)
              'predicted_return' - calibrated return prediction
              'confidence'      - isotonic-calibrated probability [0,1]

        --- ML SIDE: THE INFERENCE PIPELINE ---
        Step 1: LightGBM forward pass → raw prediction score
                This is a weighted sum of leaf values across all trees.
                For a regression model: raw score ≈ predicted return.
                For a classifier: raw score → sigmoid → probability.

        Step 2: Confidence calibration via IsotonicRegression
                Raw model confidence is often miscalibrated — a model
                that says 80% confidence might only be right 60% of
                the time. IsotonicRegression is a non-parametric
                calibration method (better than Platt scaling for
                most ML models). It fits a monotone function mapping
                raw scores to actual outcome frequencies.

        Step 3: Cross-sectional ranking
                Sort all predictions → assign ranks 1 to N.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Ensure features are in the same order as during training
        if self.metadata.feature_names:
            missing = set(self.metadata.feature_names) - set(features.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")
            features = features[self.metadata.feature_names]

        start = time.perf_counter()

        # ── Step 1: LightGBM forward pass ────────────────────────────────
        # predict() returns raw scores for regression, probabilities for
        # binary classification if num_class=1 and objective='binary'
        raw_scores = self.model.predict(features.values)

        # ── Step 2: Calibration ───────────────────────────────────────────
        if self.calibrator is not None:
            # IsotonicRegression maps raw scores → calibrated probabilities
            # Input must be 1D array, output is probability in [0,1]
            confidence = np.clip(
                self.calibrator.predict(raw_scores.reshape(-1)),
                0.0, 1.0
            )
        else:
            # No calibrator: use sigmoid to squash to [0,1]
            # sigmoid(x) = 1 / (1 + e^(-x))
            confidence = 1 / (1 + np.exp(-raw_scores))

        elapsed = (time.perf_counter() - start) * 1000

        # Track metrics for the /health endpoint
        self._prediction_count += len(features)
        self._total_latency_ms += elapsed

        logger.debug(f"Inference: {len(features)} tickers in {elapsed:.2f}ms")

        return {
            "raw_scores": raw_scores,
            "predicted_return": raw_scores,   # regression output = predicted return
            "confidence": confidence,
        }

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per prediction batch."""
        if self._prediction_count == 0:
            return 0.0
        return self._total_latency_ms / max(1, self._prediction_count // 10)

    @property
    def prediction_count(self) -> int:
        return self._prediction_count

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._load_time


# ── MODULE-LEVEL SINGLETON INSTANCE ───────────────────────────────────────────
# This is the one model manager shared across all requests.
# FastAPI's dependency injection will provide this to each route handler.
model_manager = SignalModelManager()
