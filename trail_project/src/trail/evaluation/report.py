"""
Evaluation report generator.

Aggregates results across multiple models, ratios, and seeds.
Outputs:
  - CSV summary tables
  - LaTeX-formatted tables for paper
  - JSON metrics files
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from trail.utils.config import PROJECT_ROOT
from trail.utils.logging import get_logger

logger = get_logger(__name__)


def load_all_metrics(metrics_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all JSON metric files from outputs/metrics/ and combine into a DataFrame.
    """
    metrics_dir = metrics_dir or PROJECT_ROOT / "outputs/metrics"
    rows = []

    for json_file in sorted(metrics_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            data["_source_file"] = json_file.stem
            rows.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def make_comparison_table(
    df_metrics: pd.DataFrame,
    models: list[str] = None,
    ratios: list[float] = None,
    metric_keys: list[str] = None,
) -> pd.DataFrame:
    """
    Build a wide-format comparison table for the paper.

    Columns: model | fewshot_ratio | accuracy | macro_f1 | mode_share_mae | mode_share_js
    """
    if models is None:
        models = ["mnl", "xgboost", "trail"]
    if metric_keys is None:
        metric_keys = ["accuracy", "macro_f1", "mode_share_mae", "mode_share_js"]
    if ratios is None:
        ratios = [0.01, 0.05, 0.10]

    rows = []
    for model in models:
        for ratio in ratios:
            # Find matching rows (seed-averaged)
            model_cols = {
                k: f"{model}_{k}"
                for k in metric_keys
            }
            matching = df_metrics[
                (df_metrics.get("fewshot_ratio", pd.Series()) == ratio) |
                df_metrics["_source_file"].str.contains(f"ratio{ratio:.2f}", na=False)
            ]

            row = {"model": model, "fewshot_ratio": ratio}
            for k, col in model_cols.items():
                if col in matching.columns:
                    vals = matching[col].dropna()
                    row[k] = vals.mean() if len(vals) > 0 else float("nan")
                    row[f"{k}_std"] = vals.std() if len(vals) > 1 else 0.0

            rows.append(row)

    return pd.DataFrame(rows)


def to_latex_table(
    df: pd.DataFrame,
    caption: str = "Mode choice prediction results",
    label: str = "tab:main_results",
    float_fmt: str = "{:.3f}",
) -> str:
    """
    Convert a comparison DataFrame to a LaTeX tabular string.
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * (len(df.columns) - 1) + "}",
        "\\hline",
    ]

    # Header
    header = " & ".join(str(c).replace("_", "\\_") for c in df.columns) + " \\\\"
    lines.extend([header, "\\hline"])

    # Rows
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(float_fmt.format(val) if not np.isnan(val) else "-")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def save_comparison_table(
    df_metrics: pd.DataFrame,
    output_dir: Optional[Path] = None,
    save_latex: bool = True,
) -> Path:
    """
    Build and save the main comparison table as CSV (and optionally LaTeX).
    """
    output_dir = output_dir or PROJECT_ROOT / "outputs/tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    table = make_comparison_table(df_metrics)
    csv_path = output_dir / "main_results.csv"
    table.to_csv(csv_path, index=False)
    logger.info(f"Comparison table saved: {csv_path}")

    if save_latex:
        latex_str = to_latex_table(table, caption="Main results: mode choice prediction")
        tex_path = output_dir / "main_results.tex"
        tex_path.write_text(latex_str, encoding="utf-8")
        logger.info(f"LaTeX table saved: {tex_path}")

    return csv_path
