"""
Export paper-ready tables and figures from evaluation results.

Usage:
    python scripts/export_tables_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import pandas as pd

from trail.evaluation.report import load_all_metrics, make_comparison_table, to_latex_table, save_comparison_table
from trail.visualization.fewshot_curve import plot_fewshot_curve, plot_mode_share_comparison
from trail.utils.config import PROJECT_ROOT
from trail.utils.logging import get_logger

logger = get_logger(
    "export_tables_figures",
    log_file=PROJECT_ROOT / "outputs/logs/export.log",
)


def build_results_dataframe() -> pd.DataFrame:
    """
    Collate all metrics into a model × ratio × seed DataFrame for plotting.
    """
    metrics_dir = PROJECT_ROOT / "outputs/metrics"
    rows = []

    for json_file in sorted(metrics_dir.glob("*_metrics.json")):
        stem = json_file.stem.replace("_metrics", "")
        parts = stem.split("_")
        if not parts:
            continue

        model = parts[0]
        ratio = None
        seed = None
        for part in parts[1:]:
            if part.startswith("ratio"):
                try:
                    ratio = float(part.replace("ratio", ""))
                except ValueError:
                    pass
            elif part.startswith("seed"):
                try:
                    seed = int(part.replace("seed", ""))
                except ValueError:
                    pass

        with open(json_file) as f:
            data = json.load(f)

        # Normalize metric keys (strip model prefix)
        normalized = {}
        for k, v in data.items():
            stripped = k.replace(f"{model}_", "") if k.startswith(f"{model}_") else k
            normalized[stripped] = v

        rows.append({
            "model": model,
            "fewshot_ratio": ratio,
            "seed": seed,
            **normalized,
        })

    return pd.DataFrame(rows)


def main():
    logger.info("=== Exporting Tables and Figures ===")

    out_dir_tables = PROJECT_ROOT / "outputs/tables"
    out_dir_figures = PROJECT_ROOT / "outputs/figures"
    out_dir_tables.mkdir(parents=True, exist_ok=True)
    out_dir_figures.mkdir(parents=True, exist_ok=True)

    # ---- Build results DataFrame ----
    results_df = build_results_dataframe()

    if results_df.empty:
        logger.warning("No evaluation metrics found. Run evaluate.py first.")
        return

    logger.info(f"Loaded results for {results_df['model'].nunique()} models, "
                f"{results_df['fewshot_ratio'].nunique()} ratios")

    # ---- Table 1: Main comparison table (per ratio, mean ± std across seeds) ----
    models_available = [m for m in ["mnl", "xgboost", "trail"] if m in results_df["model"].unique()]
    key_metrics = ["accuracy", "macro_f1", "mode_share_mae", "mode_share_js"]

    table_rows = []
    for model in models_available:
        for ratio in sorted(results_df["fewshot_ratio"].dropna().unique()):
            sub = results_df[(results_df["model"] == model) & (results_df["fewshot_ratio"] == ratio)]
            row = {"Model": model.upper(), "Few-shot %": f"{ratio*100:.0f}%"}
            for metric in key_metrics:
                if metric in sub.columns:
                    vals = sub[metric].dropna()
                    mean = vals.mean() if len(vals) > 0 else float("nan")
                    std = vals.std() if len(vals) > 1 else 0.0
                    row[metric.replace("_", " ").title()] = f"{mean:.3f} ± {std:.3f}" if not np.isnan(mean) else "-"
            table_rows.append(row)

    if table_rows:
        table_df = pd.DataFrame(table_rows)
        csv_path = out_dir_tables / "main_results.csv"
        table_df.to_csv(csv_path, index=False)
        logger.info(f"Main table saved: {csv_path}")

        # LaTeX
        latex_str = to_latex_table(
            table_df,
            caption="Mode Choice Prediction Results (Mean $\\pm$ Std across seeds)",
            label="tab:main_results",
        )
        tex_path = out_dir_tables / "main_results.tex"
        tex_path.write_text(latex_str, encoding="utf-8")
        logger.info(f"LaTeX table saved: {tex_path}")

    # ---- Figure 1: Few-shot adaptation curves ----
    for metric in ["macro_f1", "mode_share_mae"]:
        if metric in results_df.columns:
            # Aggregate by model × ratio (mean across seeds)
            agg = results_df.groupby(["model", "fewshot_ratio"])[metric].agg(["mean", "std"]).reset_index()
            agg.columns = ["model", "fewshot_ratio", metric, f"{metric}_std"]

            fig_path = out_dir_figures / f"fewshot_curve_{metric}.pdf"
            plot_fewshot_curve(
                agg,
                metric=metric,
                models=models_available,
                title=f"Few-shot Adaptation: {metric.replace('_', ' ').title()}",
                output_path=fig_path,
            )

    # ---- Figure 2: Mode share bar chart (if prediction files exist) ----
    pred_dir = PROJECT_ROOT / "outputs/predictions"
    pred_files = list(pred_dir.glob("*.parquet"))
    if pred_files:
        # Use ratio=0.05 predictions if available
        target_ratio = "ratio0.05"
        target_files = {
            p.stem.split("_")[0]: p
            for p in pred_files
            if target_ratio in p.stem
        }

        if target_files:
            first_file = next(iter(target_files.values()))
            df_ref = pd.read_parquet(first_file)
            y_true = df_ref["main_mode"].values

            predictions = {}
            for model_name, pred_file in target_files.items():
                df_pred = pd.read_parquet(pred_file)
                valid = df_pred["predicted_mode"].fillna(-1) >= 1
                if valid.sum() > 0:
                    predictions[model_name] = df_pred.loc[valid, "predicted_mode"].values

            if predictions:
                plot_mode_share_comparison(
                    y_true[list(target_files.values()).index(first_file)
                           if isinstance(first_file, Path) else 0:],
                    predictions,
                    output_path=out_dir_figures / "mode_share_comparison.pdf",
                )

    logger.info("=== Export Complete ===")


if __name__ == "__main__":
    main()
