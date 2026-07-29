"""
training/train.py — Model training pipeline with MLflow tracking.

This script:
1. Downloads historical price data for a universe of stocks
2. Constructs a feature matrix using signals.py
3. Labels the data (next-day return as target)
4. Trains LightGBM with walk-forward cross-validation
5. Fits an isotonic calibrator on validation predictions
6. Logs everything to MLflow (parameters, metrics, model artefacts)
7. Saves the model and metadata to disk for the API to load

Run this before starting the API:
    python -m training.train

--- WALK-FORWARD VALIDATION: WHY IT MATTERS ---
Standard k-fold cross-validation randomly splits data into train/test.
For time-series, this is WRONG — it allows future data into the
training set (look-ahead bias). Walk-forward (or expanding window)
validation respects temporal order:

  Train: 2020-01-01 → 2021-12-31  |  Val: 2022-01-01 → 2022-03-31
  Train: 2020-01-01 → 2022-03-31  |  Val: 2022-04-01 → 2022-06-30
  Train: 2020-01-01 → 2022-06-30  |  Val: 2022-07-01 → 2022-09-30
  ...

The training window EXPANDS over time. Validation always uses data
that came AFTER the training window. This is the only honest way to
evaluate time-series models.
"""

import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# Add parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.signals import fetch_price_data, compute_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# The universe of stocks to train on — S&P 500 large caps + some tech
# In production you'd use a proper index constituent list
TRAINING_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "JNJ", "V", "PG", "MA", "NFLX",
    "ADBE", "CRM", "INTC", "AMD", "QCOM", "TXN", "AVGO",
    "BAC", "WFC", "GS", "AXP", "SPGI",
    "XOM", "CVX", "SLB",
    "PFE", "MRK", "ABBV", "AMGN",
    "WMT", "COST", "KO", "PEP", "MCD",
]

# LightGBM hyperparameters
# These are tuned for financial cross-sectional prediction
LGBM_PARAMS = {
    "objective": "regression",      # predicting a continuous return
    "metric": "rmse",               # root mean squared error
    "num_leaves": 31,               # max leaves per tree (controls complexity)
    "learning_rate": 0.05,          # step size — smaller = more trees needed but better
    "feature_fraction": 0.8,        # fraction of features per tree (reduces overfitting)
    "bagging_fraction": 0.8,        # fraction of data per tree (reduces overfitting)
    "bagging_freq": 5,              # apply bagging every 5 iterations
    "min_child_samples": 20,        # min samples in a leaf (prevents overfitting)
    "reg_alpha": 0.1,               # L1 regularisation
    "reg_lambda": 0.1,              # L2 regularisation
    "verbose": -1,                  # suppress LightGBM output
    "n_jobs": -1,                   # use all CPU cores
    "random_state": 42,
}

NUM_BOOST_ROUND = 200       # number of boosting rounds (trees)
EARLY_STOPPING_ROUNDS = 20  # stop if validation metric doesn't improve for 20 rounds

# Walk-forward parameters
TRAIN_WINDOW_DAYS = 504     # 2 years of training data
VAL_WINDOW_DAYS = 63        # 3 months of validation data
N_FOLDS = 4                 # number of walk-forward folds

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ── DATA PREPARATION ──────────────────────────────────────────────────────────

def prepare_panel_data(
    tickers: list[str],
    lookback_days: int = 756
) -> pd.DataFrame:
    """
    Build a panel dataset: rows = (date, ticker), columns = features + target.

    The target (label) is the NEXT DAY's return — we're predicting what
    will happen tomorrow based on signals computed from data up to today.

    --- ML SIDE: LABELLING STRATEGY ---
    Target = next-day log return for each ticker.
    This is a "point-in-time" label: we compute today's signal, and
    we label it with tomorrow's actual return. When training, we only
    use labels for dates where we have a COMPLETE next-day observation
    (i.e. we can't label today because we don't know tomorrow yet).

    In practice: at inference time (real-time prediction), we compute
    today's signals and predict tomorrow's return — this is the API's job.
    During training, we use historical dates where tomorrow is known.
    """
    logger.info(f"Fetching {lookback_days} days of data for {len(tickers)} tickers")
    prices, failed = fetch_price_data(tickers, lookback_days=lookback_days)

    if failed:
        logger.warning(f"Skipping {len(failed)} failed tickers: {failed}")

    # Compute log returns for all tickers
    log_returns = np.log(prices / prices.shift(1))

    # Build panel: for each date, compute signals and label with next-day return
    # We iterate over each date as the "as-of" date
    rows = []
    dates = prices.index[130:]  # skip first 130 days (need history for 6mo momentum)

    for i, as_of_date in enumerate(dates[:-1]):  # exclude last date (no tomorrow)
        # Price data available up to and including as_of_date
        prices_to_date = prices.loc[:as_of_date]
        if len(prices_to_date) < 130:
            continue

        # Compute features using only data up to as_of_date
        features = compute_feature_matrix(prices_to_date)
        if features.empty:
            continue

        # Labels: next-day returns (the day AFTER as_of_date)
        next_date = dates[i + 1]
        next_returns = log_returns.loc[next_date]

        for ticker in features.index:
            if ticker in next_returns.index and not pd.isna(next_returns[ticker]):
                row = features.loc[ticker].to_dict()
                row["target"] = next_returns[ticker]
                row["date"] = as_of_date
                row["ticker"] = ticker
                rows.append(row)

        if i % 20 == 0:
            logger.info(f"Processed {i}/{len(dates)-1} dates")

    panel = pd.DataFrame(rows)
    logger.info(f"Panel dataset: {len(panel)} samples, {len(panel.columns)} columns")
    return panel


# ── WALK-FORWARD CROSS-VALIDATION ─────────────────────────────────────────────

def walk_forward_cv(
    panel: pd.DataFrame,
    n_folds: int = N_FOLDS,
) -> tuple[lgb.Booster, IsotonicRegression, dict]:
    """
    Walk-forward cross-validation with LightGBM.

    Returns:
        model: trained LightGBM model (on full data)
        calibrator: isotonic regression calibrator
        metrics: dict of validation metrics across folds

    --- ML SIDE: WHAT LGBM IS DOING ---
    LightGBM is a gradient boosting framework. It builds an ensemble
    of decision trees SEQUENTIALLY, where each new tree corrects the
    errors of the previous trees.

    Gradient boosting minimises a loss function (here: MSE) by:
    1. Start with a trivial model (predict the mean)
    2. Compute residuals (actual - predicted)
    3. Fit a tree to the residuals (learn to correct the errors)
    4. Add the tree to the ensemble (scaled by learning_rate)
    5. Repeat NUM_BOOST_ROUND times

    Key hyperparameters and why they matter:
    - num_leaves: controls tree complexity. Higher = more complex = more overfit risk
    - learning_rate: step size. Lower = need more trees but generalises better
    - feature_fraction: random subset of features per tree (like Random Forest)
    - min_child_samples: at least 20 samples in each leaf — prevents memorising noise
    - reg_alpha/lambda: L1/L2 regularisation — penalises large leaf values
    """
    feature_cols = [
        "momentum_1m", "momentum_3m", "momentum_6m",
        "mean_reversion", "volatility", "vol_momentum"
    ]
    target_col = "target"

    dates = pd.to_datetime(panel["date"]).sort_values().unique()
    total_days = len(dates)

    fold_size = (total_days - TRAIN_WINDOW_DAYS) // n_folds
    if fold_size <= 0:
        logger.warning("Not enough data for walk-forward CV. Using single split.")
        fold_size = total_days // 5

    fold_metrics = []
    all_val_preds = []
    all_val_targets = []
    best_model = None

    for fold in range(n_folds):
        train_end_idx = TRAIN_WINDOW_DAYS + fold * fold_size
        val_end_idx = train_end_idx + VAL_WINDOW_DAYS

        if val_end_idx > total_days:
            logger.info(f"Fold {fold+1}: not enough data remaining, stopping")
            break

        train_dates = dates[:train_end_idx]
        val_dates = dates[train_end_idx:val_end_idx]

        train_mask = panel["date"].isin(train_dates)
        val_mask = panel["date"].isin(val_dates)

        X_train = panel.loc[train_mask, feature_cols].values
        y_train = panel.loc[train_mask, target_col].values
        X_val = panel.loc[val_mask, feature_cols].values
        y_val = panel.loc[val_mask, target_col].values

        logger.info(
            f"Fold {fold+1}/{n_folds} | "
            f"Train: {len(X_train)} samples | "
            f"Val: {len(X_val)} samples"
        )

        # Build LightGBM datasets
        # Dataset is LightGBM's internal data structure — more memory-efficient
        # than numpy arrays for large datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train with early stopping
        # early_stopping callback stops training if val RMSE doesn't improve
        # for EARLY_STOPPING_ROUNDS iterations — prevents overfitting
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=50),
        ]

        model = lgb.train(
            params=LGBM_PARAMS,
            train_set=train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Evaluate on validation set
        val_preds = model.predict(X_val)

        # IC (Information Coefficient) = Spearman rank correlation
        # between predicted and actual returns
        # This is the same IC metric from your signal research framework
        ic, _ = spearmanr(val_preds, y_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))

        # Sharpe-like metric: IC × sqrt(annual_breadth) / IC_std
        # approximation since we don't have full portfolio construction here
        ic_per_date = []
        val_panel = panel.loc[val_mask].copy()
        val_panel["pred"] = val_preds
        for date, grp in val_panel.groupby("date"):
            if len(grp) >= 5:
                ic_d, _ = spearmanr(grp["pred"], grp[target_col])
                if not np.isnan(ic_d):
                    ic_per_date.append(ic_d)

        ic_mean = np.mean(ic_per_date) if ic_per_date else 0.0
        icir = (np.mean(ic_per_date) / np.std(ic_per_date)) if len(ic_per_date) > 1 else 0.0

        fold_metrics.append({
            "fold": fold + 1,
            "ic": float(ic),
            "ic_mean_daily": float(ic_mean),
            "icir": float(icir),
            "rmse": float(rmse),
            "best_iteration": model.best_iteration,
        })

        logger.info(
            f"Fold {fold+1} | IC={ic:.4f} | ICIR={icir:.2f} | RMSE={rmse:.6f}"
        )

        all_val_preds.extend(val_preds.tolist())
        all_val_targets.extend(y_val.tolist())
        best_model = model  # keep last fold model

    # ── CALIBRATION ───────────────────────────────────────────────────────────
    # Fit isotonic regression calibrator on all out-of-sample validation predictions
    # Maps raw LightGBM scores → calibrated probabilities

    logger.info("Fitting isotonic calibration on OOS validation predictions")
    calibrator = IsotonicRegression(out_of_bounds="clip")

    # Convert regression targets to binary labels for calibration
    # 1 = positive return (stock went up), 0 = negative return
    binary_targets = (np.array(all_val_targets) > 0).astype(float)

    # Normalise raw scores to [0,1] range first using sigmoid
    sigmoid_preds = 1 / (1 + np.exp(-np.array(all_val_preds)))
    calibrator.fit(sigmoid_preds, binary_targets)

    # Aggregate metrics across folds
    avg_metrics = {
        "avg_ic": float(np.mean([m["ic"] for m in fold_metrics])),
        "avg_icir": float(np.mean([m["icir"] for m in fold_metrics])),
        "avg_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "n_folds": len(fold_metrics),
        "fold_details": fold_metrics,
    }

    logger.info(
        f"Walk-forward complete | "
        f"Avg IC={avg_metrics['avg_ic']:.4f} | "
        f"Avg ICIR={avg_metrics['avg_icir']:.2f}"
    )

    return best_model, calibrator, avg_metrics


# ── MLFLOW LOGGING ────────────────────────────────────────────────────────────

def train_and_log():
    """
    Full training run with MLflow experiment tracking.

    MLflow tracks:
    - Parameters: hyperparameters used for this run
    - Metrics: IC, ICIR, RMSE across folds
    - Artefacts: the trained model file, calibrator, metadata

    This means you can compare multiple runs (different hyperparameters,
    different universes) in the MLflow UI and know exactly which model
    is deployed and how it was trained.

    Run `mlflow ui` in the project root to view the experiment browser.
    """
    # Set up MLflow experiment
    # An "experiment" is a named collection of runs
    mlflow.set_experiment("ml-signal-api")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # ── Log hyperparameters ───────────────────────────────────────────
        # Everything that affects the model is logged so you can
        # reproduce this exact run later
        mlflow.log_params({
            **LGBM_PARAMS,
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "train_window_days": TRAIN_WINDOW_DAYS,
            "val_window_days": VAL_WINDOW_DAYS,
            "n_folds": N_FOLDS,
            "universe_size": len(TRAINING_UNIVERSE),
        })

        # ── Prepare data ──────────────────────────────────────────────────
        panel = prepare_panel_data(TRAINING_UNIVERSE, lookback_days=756)
        mlflow.log_metric("training_samples", len(panel))

        # ── Walk-forward CV ───────────────────────────────────────────────
        model, calibrator, metrics = walk_forward_cv(panel)

        # ── Log validation metrics ────────────────────────────────────────
        mlflow.log_metric("avg_ic", metrics["avg_ic"])
        mlflow.log_metric("avg_icir", metrics["avg_icir"])
        mlflow.log_metric("avg_rmse", metrics["avg_rmse"])
        for fold_detail in metrics["fold_details"]:
            fold_n = fold_detail["fold"]
            mlflow.log_metric(f"fold_{fold_n}_ic", fold_detail["ic"])
            mlflow.log_metric(f"fold_{fold_n}_icir", fold_detail["icir"])

        # ── Save model artefacts to disk ─────────────────────────────────
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

        joblib.dump(model, MODEL_DIR / "signal_model.pkl")
        joblib.dump(calibrator, MODEL_DIR / "calibrator.pkl")

        feature_cols = [
            "momentum_1m", "momentum_3m", "momentum_6m",
            "mean_reversion", "volatility", "vol_momentum"
        ]
        metadata = {
            "version": version,
            "trained_at": datetime.now().isoformat(),
            "training_tickers": len(TRAINING_UNIVERSE),
            "training_days": TRAIN_WINDOW_DAYS,
            "val_sharpe": metrics["avg_icir"],  # proxy Sharpe via ICIR
            "val_ic": metrics["avg_ic"],
            "feature_names": feature_cols,
            "mlflow_run_id": run_id,
        }

        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Log artefacts to MLflow (stored alongside the run)
        mlflow.log_artifact(str(MODEL_DIR / "signal_model.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "calibrator.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "metadata.json"))

        logger.info(f"Model saved | version={version}")
        logger.info(f"MLflow run complete: {run_id}")
        logger.info(f"View UI: mlflow ui --port 5000")

        return metadata


if __name__ == "__main__":
    train_and_log()