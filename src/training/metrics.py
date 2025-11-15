"""Evaluation metrics for regression tasks."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: Ground truth values (n_samples, n_targets)
        y_pred: Predicted values (n_samples, n_targets)

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    # Handle single or multi-output regression
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_true.shape[1]

    # Compute metrics for each target
    for i in range(n_targets):
        prefix = f"target_{i}_" if n_targets > 1 else ""

        # Mean Squared Error
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        metrics[f"{prefix}mse"] = mse

        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        metrics[f"{prefix}rmse"] = rmse

        # Mean Absolute Error
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        metrics[f"{prefix}mae"] = mae

        # R² Score
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        metrics[f"{prefix}r2"] = r2

        # Mean Absolute Percentage Error (avoid division by zero)
        mask = y_true[:, i] != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask, i] - y_pred[mask, i]) / y_true[mask, i])) * 100
            metrics[f"{prefix}mape"] = mape

    # Overall metrics (average across targets if multi-output)
    if n_targets > 1:
        metrics["avg_mse"] = np.mean([metrics[f"target_{i}_mse"] for i in range(n_targets)])
        metrics["avg_rmse"] = np.mean([metrics[f"target_{i}_rmse"] for i in range(n_targets)])
        metrics["avg_mae"] = np.mean([metrics[f"target_{i}_mae"] for i in range(n_targets)])
        metrics["avg_r2"] = np.mean([metrics[f"target_{i}_r2"] for i in range(n_targets)])

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

    for name, value in metrics.items():
        if "r2" in name:
            print(f"{name:20s}: {value:8.4f}")
        elif "mape" in name:
            print(f"{name:20s}: {value:7.2f}%")
        else:
            print(f"{name:20s}: {value:8.4f}")

    print(f"{'=' * 50}\n")
