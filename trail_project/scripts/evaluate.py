"""
Evaluate all model predictions from the outputs/predictions/ directory.

Reads parquet prediction files and computes micro + macro metrics,
saving results to outputs/metrics/.

Usage:
    python scripts/evaluate.py --task mode_choice
    python scripts/evaluate.py --model trail --fewshot_ratio 0.01
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from trail.evaluation.micro import compute_micro_metrics
from trail.evaluation.macro import compute_macro_metrics, compute_subgroup_macro
from trail.utils.config import PROJECT_ROOT
from trail.utils.logging import get_logger

logger = get_logger(
    "evaluate",
    log_file=PROJECT_ROOT / "outputs/logs/evaluate.log",
)


def evaluate_prediction_file(
    pred_path: Path,
    model_name: str,
    target_col: str = "main_mode",
    pred_col: str = "predicted_mode",
    subgroups: list[str] = None,
) -> dict:
    """Load a prediction parquet file and compute all metrics."""
    df = pd.read_parquet(pred_path)
    logger.info(f"Evaluating {pred_path.stem}: {len(df):,} records")

    # Filter to valid predictions
    valid = df[pred_col].fillna(-1) >= 1
    df_eval = df[valid].copy()
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        logger.warning(f"{n_invalid} invalid predictions filtered")

    y_true = df_eval[target_col].values
    y_pred = df_eval[pred_col].values

    micro = compute_micro_metrics(y_true, y_pred, prefix=f"{model_name}_")
    macro = compute_macro_metrics(
        df_eval,
        predictions_col=pred_col,
        target_col=target_col,
        prefix=f"{model_name}_",
    )

    # Subgroup analysis
    subgroups = subgroups or ["age_group", "sex", "employment_status"]
    subgroup_results = {}
    for sg in subgroups:
        if sg in df_eval.columns:
            sg_df = compute_subgroup_macro(df_eval, pred_col, target_col, subgroup_col=sg)
            subgroup_results[sg] = sg_df.to_dict(orient="records")

    metrics = {**micro, **macro, "_n_valid": int(valid.sum()), "_n_total": len(df)}

    # Save per-file JSON
    out_path = PROJECT_ROOT / "outputs/metrics" / f"{pred_path.stem}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if subgroup_results:
        sg_path = out_path.with_suffix(".subgroups.json")
        with open(sg_path, "w") as f:
            json.dump(subgroup_results, f, indent=2)

    return metrics


def main(args):
    pred_dir = PROJECT_ROOT / "outputs/predictions"
    if not pred_dir.exists():
        logger.warning("No predictions directory found.")
        return

    parquet_files = sorted(pred_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning("No prediction files found in outputs/predictions/")
        return

    all_results = []

    for pred_file in parquet_files:
        # Parse model name from filename: {model}_ratio{r}_seed{s}.parquet
        stem = pred_file.stem
        parts = stem.split("_")
        model_name = parts[0]

        if args.model and model_name != args.model:
            continue

        try:
            metrics = evaluate_prediction_file(
                pred_file,
                model_name=model_name,
                target_col=args.target_col,
                pred_col=args.pred_col,
            )
            metrics["_file"] = stem
            metrics["_model"] = model_name
            all_results.append(metrics)

            acc_key = f"{model_name}_accuracy"
            f1_key = f"{model_name}_macro_f1"
            mae_key = f"{model_name}_mode_share_mae"
            logger.info(
                f"  acc={metrics.get(acc_key, metrics.get('accuracy', 0)):.3f} "
                f"f1={metrics.get(f1_key, metrics.get('macro_f1', 0)):.3f} "
                f"mae={metrics.get(mae_key, metrics.get('mode_share_mae', 0)):.4f}"
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {pred_file}: {e}")

    if all_results:
        summary = pd.DataFrame(all_results)
        summary_path = PROJECT_ROOT / "outputs/metrics/evaluation_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"Summary saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="mode_choice")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to specific model name")
    parser.add_argument("--fewshot_ratio", type=float, default=None)
    parser.add_argument("--target_col", type=str, default="main_mode")
    parser.add_argument("--pred_col", type=str, default="predicted_mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
