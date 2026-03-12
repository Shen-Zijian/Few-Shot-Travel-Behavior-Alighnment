"""
Micro-level evaluation metrics (individual prediction quality).

Metrics:
  - accuracy
  - macro_f1, weighted_f1
  - per-class precision, recall, f1
  - NLL (negative log-likelihood)
  - Brier score
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    log_loss,
)

from trail.data.schema import MODE_LABELS
from trail.utils.logging import get_logger

logger = get_logger(__name__)


def compute_micro_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: Optional[np.ndarray] = None,
    label_names: Optional[dict] = None,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute standard micro-level metrics.

    Args:
        y_true: ground truth mode labels
        y_pred: predicted mode labels
        y_proba: predicted probability matrix (N x K); required for NLL/Brier
        label_names: mapping from int label to string name
        prefix: string prefix for metric keys (e.g. "mnl_")

    Returns:
        dict of metric_name -> value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    label_names = label_names or MODE_LABELS

    classes = np.sort(np.unique(y_true))
    metrics: dict[str, float] = {}

    metrics[f"{prefix}accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics[f"{prefix}macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics[f"{prefix}weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class F1
    per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    for cls, f1 in zip(classes, per_class):
        name = label_names.get(cls, str(cls))
        metrics[f"{prefix}f1_{name.replace(' ', '_')}"] = float(f1)

    # Probability-based metrics
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        # Map labels to 0-indexed for sklearn
        label_list = sorted(label_names.keys())
        label_to_idx = {c: i for i, c in enumerate(label_list)}

        y_true_idx = np.array([label_to_idx.get(c, 0) for c in y_true])
        proba_aligned = _align_proba(y_proba, classes, label_list)

        try:
            metrics[f"{prefix}nll"] = float(log_loss(y_true_idx, proba_aligned))
        except Exception:
            metrics[f"{prefix}nll"] = float("nan")

        metrics[f"{prefix}brier_score"] = float(_brier_score_multiclass(y_true_idx, proba_aligned))

    return metrics


def _align_proba(y_proba: np.ndarray, model_classes: np.ndarray, target_labels: list) -> np.ndarray:
    """
    Align a probability matrix (columns = model_classes) to target_labels ordering.
    Fills zeros for labels not predicted by the model.
    """
    n = len(y_proba)
    k = len(target_labels)
    aligned = np.zeros((n, k), dtype=float)
    for col_idx, cls in enumerate(model_classes):
        if col_idx < y_proba.shape[1] and cls in target_labels:
            tgt_idx = target_labels.index(cls)
            aligned[:, tgt_idx] = y_proba[:, col_idx]
    # Normalize rows to sum to 1
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return aligned / row_sums


def _brier_score_multiclass(y_true_idx: np.ndarray, y_proba: np.ndarray) -> float:
    """Multi-class Brier score = mean squared error of probability predictions."""
    n, k = y_proba.shape
    one_hot = np.zeros_like(y_proba)
    for i, label in enumerate(y_true_idx):
        if 0 <= label < k:
            one_hot[i, label] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def evaluate_predictions(
    df_test: pd.DataFrame,
    predictions_col: str = "predicted_mode",
    proba_cols: Optional[list[str]] = None,
    target_col: str = "main_mode",
    model_name: str = "",
) -> dict[str, float]:
    """
    Evaluate predictions stored in a DataFrame.

    df_test must contain: target_col and predictions_col.
    """
    y_true = df_test[target_col].values
    y_pred = df_test[predictions_col].values

    y_proba = None
    if proba_cols is not None:
        y_proba = df_test[proba_cols].values

    prefix = f"{model_name}_" if model_name else ""
    metrics = compute_micro_metrics(y_true, y_pred, y_proba, prefix=prefix)

    logger.info(
        f"[{model_name}] Accuracy={metrics.get(f'{prefix}accuracy', 0):.3f} "
        f"MacroF1={metrics.get(f'{prefix}macro_f1', 0):.3f}"
    )
    return metrics
