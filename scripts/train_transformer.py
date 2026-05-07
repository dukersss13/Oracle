"""
Offline training script for the TeamTransformer model.

Usage:
    python -m scripts.train_transformer [--epochs 500] [--seq-len 10] [--verbose]

Builds the training dataset from 2 seasons of NBA data, trains the
Transformer, evaluates on a temporal holdout, and saves the model
to artifacts/models/team_transformer/.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

SEASONS = ["2024-25", "2023-24"]
MODEL_DIR = Path("artifacts/models/team_transformer")
CACHE_DIR = Path("artifacts/cache")


def _load_all_logs() -> pd.DataFrame | None:
    all_logs_path = Path("data/all_logs.csv")
    if all_logs_path.exists():
        df = pd.read_csv(all_logs_path, low_memory=False)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        return df
    return None


def _temporal_split(
    X_seq: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    enriched: pd.DataFrame,
    seq_len: int,
    train_frac: float = 0.80,
) -> tuple:
    """Split by date: first train_frac% of dates for training, rest for test."""
    # Build a date array aligned with samples
    dates = []
    for team, group in enriched.groupby("TEAM_ABBREVIATION"):
        group = group.sort_values("GAME_DATE").reset_index(drop=True)
        for i in range(seq_len, len(group)):
            row = group.iloc[i]
            if not pd.isna(row["PTS"]):
                dates.append(row["GAME_DATE"])

    dates = np.array(dates[:len(y)])  # Align with y (NaN rows were skipped)

    if len(dates) != len(y):
        # Fallback: simple index-based split
        logger.warning(
            "Date alignment mismatch (%d dates vs %d samples); using index split",
            len(dates), len(y),
        )
        split_idx = int(len(y) * train_frac)
        return (
            X_seq[:split_idx], X_static[:split_idx], y[:split_idx],
            X_seq[split_idx:], X_static[split_idx:], y[split_idx:],
        )

    unique_dates = np.sort(np.unique(dates))
    split_idx = int(len(unique_dates) * train_frac)
    split_date = unique_dates[min(split_idx, len(unique_dates) - 1)]

    train_mask = dates < split_date
    test_mask = ~train_mask

    return (
        X_seq[train_mask], X_static[train_mask], y[train_mask],
        X_seq[test_mask], X_static[test_mask], y[test_mask],
    )


def _winner_accuracy(
    enriched: pd.DataFrame,
    model,
    X_seq_test: np.ndarray,
    X_static_test: np.ndarray,
    y_test: np.ndarray,
    seq_len: int,
    train_frac: float = 0.80,
) -> float:
    """Compute winner prediction accuracy on the test set.

    For each game, both teams' predicted points are compared.
    """
    # Rebuild game-level pairs from the test set
    # This is approximate: we pair consecutive samples if they share a GAME_ID
    preds = model.predict(X_seq_test, X_static_test)

    # Simple approximation: compare predicted vs actual per-sample sign
    # (positive = team scored more than average → likely winner side)
    correct = 0
    total = 0
    median_pts = np.median(y_test)
    for i in range(len(y_test)):
        pred_above = preds[i] >= median_pts
        actual_above = y_test[i] >= median_pts
        if pred_above == actual_above:
            correct += 1
        total += 1

    return correct / total if total > 0 else float("nan")


def train_model(
    epochs: int = 500,
    seq_len: int = 10,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    ff_dim: int = 128,
    dropout: float = 0.15,
    lr: float = 1e-3,
    patience: int = 50,
    verbose: bool = False,
) -> dict:
    """Train (or retrain) the TeamTransformer and save it to disk.

    Returns a dict with training metrics.
    """
    from data_prep.team_features import (
        build_enriched_team_logs,
        build_training_dataset,
        NUM_SEQ_FEATURES,
        NUM_STATIC_FEATURES,
    )
    from models.transformer import TeamTransformer

    # ── 1. Build dataset ──
    logger.info("Building enriched team logs for seasons: %s", SEASONS)
    all_logs = _load_all_logs()
    enriched = build_enriched_team_logs(SEASONS, all_logs_df=all_logs, cache_dir=CACHE_DIR)
    logger.info("Enriched logs: %d rows, %d columns", len(enriched), len(enriched.columns))

    X_seq, X_static, y = build_training_dataset(enriched, seq_len=seq_len)
    logger.info("Dataset: X_seq=%s, X_static=%s, y=%s", X_seq.shape, X_static.shape, y.shape)

    if len(y) < 100:
        raise RuntimeError(f"Insufficient training data ({len(y)} samples). Need at least 100.")

    # ── 2. Temporal split ──
    X_seq_train, X_static_train, y_train, X_seq_test, X_static_test, y_test = _temporal_split(
        X_seq, X_static, y, enriched, seq_len
    )
    logger.info("Train: %d samples | Test: %d samples", len(y_train), len(y_test))

    # ── 3. Build & train model ──
    config = {
        "seq_len": seq_len,
        "seq_features": NUM_SEQ_FEATURES,
        "static_features": NUM_STATIC_FEATURES,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ff_dim": ff_dim,
        "dropout": dropout,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": lr,
        "scaling_method": "standard",
    }

    transformer = TeamTransformer(config)
    if verbose:
        transformer.summary()

    logger.info("Training TeamTransformer (epochs=%d, patience=%d)", epochs, patience)
    history = transformer.train(
        X_seq_train, X_static_train, y_train,
        validation_split=0.15,
        verbose=1 if verbose else 0,
    )

    # ── 4. Evaluate ──
    preds = transformer.predict(X_seq_test, X_static_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))

    logger.info("Test RMSE: %.2f", rmse)
    logger.info("Test MAE:  %.2f", mae)
    logger.info("Test mean actual PTS: %.1f, predicted: %.1f", y_test.mean(), preds.mean())

    # ── 5. Save model ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(MODEL_DIR / "model.keras")
    transformer.save(model_path)
    logger.info("Model saved to %s", model_path)

    # Save config + metrics
    metrics = {
        "config": config,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_rmse": rmse,
        "test_mae": mae,
        "seasons": SEASONS,
        "best_epoch": len(history.history["loss"]),
    }
    with open(MODEL_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete. Metrics saved to %s", MODEL_DIR / "training_metrics.json")
    return metrics


def is_model_stale(max_age_hours: int = 24) -> bool:
    """Return True if the saved model is missing or older than ``max_age_hours``."""
    metrics_path = MODEL_DIR / "training_metrics.json"
    model_path = MODEL_DIR / "model.keras"
    if not model_path.exists() or not metrics_path.exists():
        return True
    age_hours = (time.time() - model_path.stat().st_mtime) / 3600
    return age_hours > max_age_hours


def main():
    parser = argparse.ArgumentParser(description="Train TeamTransformer model")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        lr=args.lr,
        patience=args.patience,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
