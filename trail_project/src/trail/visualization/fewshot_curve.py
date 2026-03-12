"""
Few-shot adaptation curve visualization.

Plots model performance (Macro-F1 and mode share MAE) vs. few-shot ratio
for all compared models. Produces paper-ready figures.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trail.utils.config import PROJECT_ROOT
from trail.utils.logging import get_logger

logger = get_logger(__name__)

# Paper-quality settings
FIGSIZE = (10, 4)
DPI = 300
FONT_SIZE = 11

MODEL_STYLES = {
    "mnl": {"color": "#2196F3", "linestyle": "--", "marker": "s", "label": "MNL"},
    "xgboost": {"color": "#4CAF50", "linestyle": "-.", "marker": "^", "label": "XGBoost"},
    "prompt_only": {"color": "#FF9800", "linestyle": ":", "marker": "D", "label": "Prompt-Only LLM"},
    "trail": {"color": "#E91E63", "linestyle": "-", "marker": "o", "label": "TRAIL (Ours)"},
}


def plot_fewshot_curve(
    results_df: pd.DataFrame,
    metric: str = "macro_f1",
    models: Optional[list[str]] = None,
    title: str = "",
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Plot performance vs. few-shot ratio for multiple models.

    Args:
        results_df: DataFrame with columns [model, fewshot_ratio, {metric}, {metric}_std]
        metric: metric column to plot (macro_f1, mode_share_mae, etc.)
        models: list of model names to include
        title: figure title
        output_path: save path (defaults to outputs/figures/)
        show: display interactively

    Returns:
        path to saved figure
    """
    if models is None:
        models = [m for m in MODEL_STYLES if m in results_df["model"].unique()]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    for model in models:
        style = MODEL_STYLES.get(model, {"color": "gray", "linestyle": "-", "marker": "o", "label": model})
        model_data = results_df[results_df["model"] == model].sort_values("fewshot_ratio")

        if model_data.empty:
            continue

        x = model_data["fewshot_ratio"].values * 100  # Convert to percentage
        y = model_data[metric].values
        y_std = model_data.get(f"{metric}_std", pd.Series(np.zeros_like(y))).values

        ax.plot(
            x, y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2,
            markersize=7,
            label=style["label"],
        )

        if y_std is not None and np.any(y_std > 0):
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.15, color=style["color"])

    ax.set_xlabel("Few-shot ratio (%)", fontsize=FONT_SIZE)
    metric_label = _metric_display_name(metric)
    ax.set_ylabel(metric_label, fontsize=FONT_SIZE)

    if title:
        ax.set_title(title, fontsize=FONT_SIZE + 1)

    ax.legend(fontsize=FONT_SIZE - 1, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE - 1)

    if output_path is None:
        out_dir = PROJECT_ROOT / "outputs/figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"fewshot_curve_{metric}.pdf"

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Few-shot curve saved: {output_path}")

    if show:
        plt.show()

    return output_path


def plot_mode_share_comparison(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    weights: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Bar chart comparing true vs predicted mode shares for each model.
    """
    from trail.evaluation.macro import compute_mode_shares
    from trail.data.schema import MODE_LABELS

    all_modes = sorted(MODE_LABELS.keys())
    mode_names = [MODE_LABELS[m] for m in all_modes]

    true_shares = compute_mode_shares(y_true, weights, all_modes)
    true_arr = np.array([true_shares[m] for m in all_modes])

    n_models = len(predictions)
    x = np.arange(len(all_modes))
    width = 0.7 / (n_models + 1)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI)

    # True bars
    ax.bar(x, true_arr, width=width, label="Ground Truth", color="#455A64", alpha=0.8)

    for i, (model_name, y_pred) in enumerate(predictions.items()):
        style = MODEL_STYLES.get(model_name, {"color": "gray", "label": model_name})
        pred_shares = compute_mode_shares(y_pred, None, all_modes)
        pred_arr = np.array([pred_shares[m] for m in all_modes])
        offset = (i + 1) * width
        ax.bar(x + offset, pred_arr, width=width, label=style["label"], color=style["color"], alpha=0.8)

    ax.set_xticks(x + width * n_models / 2)
    ax.set_xticklabels(mode_names, rotation=30, ha="right", fontsize=FONT_SIZE - 1)
    ax.set_ylabel("Mode Share", fontsize=FONT_SIZE)
    ax.set_title("Mode Share: Ground Truth vs Predictions", fontsize=FONT_SIZE + 1)
    ax.legend(fontsize=FONT_SIZE - 1)
    ax.grid(axis="y", alpha=0.3)

    if output_path is None:
        out_dir = PROJECT_ROOT / "outputs/figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "mode_share_comparison.pdf"

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Mode share comparison saved: {output_path}")
    return output_path


def _metric_display_name(metric: str) -> str:
    names = {
        "macro_f1": "Macro-F1",
        "accuracy": "Accuracy",
        "mode_share_mae": "Mode Share MAE",
        "mode_share_js": "JS Divergence",
        "nll": "NLL",
        "brier_score": "Brier Score",
    }
    return names.get(metric, metric.replace("_", " ").title())
