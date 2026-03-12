"""
Macro-level (population-level) evaluation metrics.

These metrics assess how well the predicted distribution matches
the true aggregate distribution — critical for travel demand analysis.

Metrics:
  - mode_share_mae: mean absolute error of mode share
  - mode_share_js: Jensen-Shannon divergence of mode share distribution
  - Subgroup mode share gaps
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from trail.data.schema import MODE_LABELS
from trail.utils.logging import get_logger

logger = get_logger(__name__)


def compute_mode_shares(
    labels: np.ndarray | pd.Series,
    weights: Optional[np.ndarray | pd.Series] = None,
    all_modes: Optional[list[int]] = None,
) -> dict[int, float]:
    """
    Compute weighted mode shares (proportions).

    Args:
        labels: array of mode labels
        weights: optional trip weights (sum to 1 after normalization)
        all_modes: full list of mode codes (to include zero-count modes)

    Returns:
        dict {mode_code: share}
    """
    labels = np.asarray(labels)
    if all_modes is None:
        all_modes = sorted(set(labels))

    if weights is None:
        counts = {m: float(np.sum(labels == m)) for m in all_modes}
        total = float(len(labels))
    else:
        weights = np.asarray(weights, dtype=float)
        counts = {m: float(np.sum(weights[labels == m])) for m in all_modes}
        total = float(weights.sum())

    if total == 0:
        return {m: 0.0 for m in all_modes}

    return {m: counts[m] / total for m in all_modes}


def mode_share_mae(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    weights: Optional[np.ndarray | pd.Series] = None,
    all_modes: Optional[list[int]] = None,
) -> float:
    """
    Mean absolute error of mode shares.

    MAE = mean_over_modes |true_share_m - pred_share_m|
    """
    if all_modes is None:
        all_modes = sorted(set(np.concatenate([np.unique(y_true), np.unique(y_pred)])))

    true_shares = compute_mode_shares(y_true, weights, all_modes)
    pred_shares = compute_mode_shares(y_pred, None, all_modes)

    mae = np.mean([abs(true_shares[m] - pred_shares[m]) for m in all_modes])
    return float(mae)


def mode_share_js_divergence(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    weights: Optional[np.ndarray | pd.Series] = None,
    all_modes: Optional[list[int]] = None,
) -> float:
    """
    Jensen-Shannon divergence between true and predicted mode share distributions.
    Returns a value in [0, 1] (0 = identical distributions).
    """
    if all_modes is None:
        all_modes = sorted(set(np.concatenate([np.unique(y_true), np.unique(y_pred)])))

    true_shares = compute_mode_shares(y_true, weights, all_modes)
    pred_shares = compute_mode_shares(y_pred, None, all_modes)

    p = np.array([true_shares[m] for m in all_modes]) + 1e-10
    q = np.array([pred_shares[m] for m in all_modes]) + 1e-10

    p = p / p.sum()
    q = q / q.sum()

    return float(jensenshannon(p, q))


def compute_macro_metrics(
    df_test: pd.DataFrame,
    predictions_col: str = "predicted_mode",
    target_col: str = "main_mode",
    weight_col: str = "trip_weight",
    all_modes: Optional[list[int]] = None,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute all macro-level metrics from a predictions DataFrame.
    """
    y_true = df_test[target_col].values
    y_pred = df_test[predictions_col].values
    weights = df_test[weight_col].values if weight_col in df_test.columns else None

    if all_modes is None:
        all_modes = sorted(MODE_LABELS.keys())

    metrics: dict[str, float] = {}

    # Mode share MAE
    metrics[f"{prefix}mode_share_mae"] = mode_share_mae(y_true, y_pred, weights, all_modes)

    # JS divergence
    metrics[f"{prefix}mode_share_js"] = mode_share_js_divergence(y_true, y_pred, weights, all_modes)

    # Per-mode share gap
    true_shares = compute_mode_shares(y_true, weights, all_modes)
    pred_shares = compute_mode_shares(y_pred, None, all_modes)

    for mode_code in all_modes:
        mode_label = MODE_LABELS.get(mode_code, str(mode_code)).replace(" ", "_")
        gap = pred_shares[mode_code] - true_shares[mode_code]
        metrics[f"{prefix}share_gap_{mode_label}"] = float(gap)
        metrics[f"{prefix}true_share_{mode_label}"] = float(true_shares[mode_code])
        metrics[f"{prefix}pred_share_{mode_label}"] = float(pred_shares[mode_code])

    logger.info(
        f"Macro metrics: mode_share_MAE={metrics.get(f'{prefix}mode_share_mae', 0):.4f} "
        f"JS={metrics.get(f'{prefix}mode_share_js', 0):.4f}"
    )

    return metrics


def compute_subgroup_macro(
    df_test: pd.DataFrame,
    predictions_col: str = "predicted_mode",
    target_col: str = "main_mode",
    subgroup_col: str = "age_group",
    weight_col: Optional[str] = "trip_weight",
    all_modes: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute mode share MAE per subgroup.

    Returns a DataFrame with columns: [subgroup_col, subgroup_value, mode_share_mae, mode_share_js]
    """
    if all_modes is None:
        all_modes = sorted(MODE_LABELS.keys())

    rows = []
    for group_val, group_df in df_test.groupby(subgroup_col):
        y_true = group_df[target_col].values
        y_pred = group_df[predictions_col].values
        weights = group_df[weight_col].values if weight_col and weight_col in group_df.columns else None

        mae = mode_share_mae(y_true, y_pred, weights, all_modes)
        js = mode_share_js_divergence(y_true, y_pred, weights, all_modes)

        rows.append({
            subgroup_col: group_val,
            "n_trips": len(group_df),
            "mode_share_mae": mae,
            "mode_share_js": js,
        })

    return pd.DataFrame(rows)
