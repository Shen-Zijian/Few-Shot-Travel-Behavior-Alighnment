"""
Run baseline models for mode choice prediction (MVP-1).

Usage:
    python scripts/run_baseline.py --model mnl --task mode_choice --fewshot_ratio 0.01
    python scripts/run_baseline.py --model xgboost --task mode_choice --fewshot_ratio 0.05
    python scripts/run_baseline.py --model all --task mode_choice
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from trail.data.splitter import split_2022
from trail.evaluation.micro import compute_micro_metrics
from trail.evaluation.macro import compute_macro_metrics
from trail.utils.config import PROJECT_ROOT, load_experiment_config
from trail.utils.io import load_df, save_json
from trail.utils.logging import get_logger
from trail.utils.seed import set_seed

logger = get_logger(
    "run_baseline",
    log_file=PROJECT_ROOT / "outputs/logs/run_baseline.log",
)


def run_mnl(df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str) -> dict:
    from trail.baselines.mnl import MNLBaseline

    model = MNLBaseline()
    model.fit(df_train, target_col=target_col)

    y_pred = model.predict(df_test)
    y_proba = model.predict_proba(df_test)

    return {"y_pred": y_pred, "y_proba": y_proba, "classes": model.classes_}


def run_xgboost(df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str, seed: int) -> dict:
    from trail.baselines.xgboost import XGBoostBaseline

    model = XGBoostBaseline(seed=seed)
    model.fit(df_train, target_col=target_col)

    y_pred = model.predict(df_test)
    y_proba = model.predict_proba(df_test)

    return {"y_pred": y_pred, "y_proba": y_proba, "classes": model.classes_}


def evaluate_and_log(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    df_test: pd.DataFrame,
    target_col: str,
) -> dict:
    """Run micro + macro evaluation and return combined metrics dict."""
    micro = compute_micro_metrics(
        y_true, y_pred, y_proba, prefix=f"{model_name}_"
    )

    df_test_with_pred = df_test.copy()
    df_test_with_pred["predicted_mode"] = y_pred

    macro = compute_macro_metrics(
        df_test_with_pred,
        predictions_col="predicted_mode",
        target_col=target_col,
        prefix=f"{model_name}_",
    )

    return {**micro, **macro}


def main(args):
    set_seed(args.seed)

    # Load preprocessed 2022 data
    path_2022 = PROJECT_ROOT / "data/interim/harmonized/tcs2022_harmonized.parquet"
    if not path_2022.exists():
        raise FileNotFoundError(
            f"Run preprocess_2022.py first. Missing: {path_2022}"
        )

    df_2022 = load_df(path_2022)
    logger.info(f"Loaded 2022 data: {len(df_2022):,} records")

    target_col = "main_mode"

    # Run for all fewshot_ratios
    fewshot_ratios = [args.fewshot_ratio] if args.fewshot_ratio else [0.01, 0.05, 0.10]
    seeds = [args.seed]

    all_results = []

    for ratio in fewshot_ratios:
        for seed in seeds:
            split = split_2022(df_2022, test_ratio=0.2, fewshot_ratio=ratio, seed=seed)
            
            df_train = split["few_shot"]  # Baseline trains on few-shot only
            print(df_train.columns)
            df_test = split["test"]

            logger.info(
                f"Few-shot ratio={ratio:.2f} seed={seed}: "
                f"train={len(df_train):,} test={len(df_test):,}"
            )

            y_true = df_test[target_col].values
            models_to_run = (
                ["mnl", "xgboost"] if args.model == "all" else [args.model]
            )

            row = {"fewshot_ratio": ratio, "seed": seed, "n_train": len(df_train), "n_test": len(df_test)}

            if "mnl" in models_to_run:
                logger.info("Running MNL ...")
                try:
                    mnl_out = run_mnl(df_train, df_test, target_col)
                    metrics = evaluate_and_log(
                        "mnl", y_true, mnl_out["y_pred"], mnl_out["y_proba"],
                        mnl_out["classes"], df_test, target_col
                    )
                    row.update(metrics)

                    # Save predictions
                    pred_path = (
                        PROJECT_ROOT / f"outputs/predictions/mnl_ratio{ratio:.2f}_seed{seed}.parquet"
                    )
                    df_pred = df_test.copy()
                    df_pred["predicted_mode"] = mnl_out["y_pred"]
                    df_pred.to_parquet(pred_path, index=False)
                    logger.info(f"MNL acc={metrics.get('mnl_accuracy', 0):.3f} f1={metrics.get('mnl_macro_f1', 0):.3f}")
                except Exception as e:
                    logger.error(f"MNL failed: {e}")

            if "xgboost" in models_to_run:
                logger.info("Running XGBoost ...")
                try:
                    xgb_out = run_xgboost(df_train, df_test, target_col, seed)
                    metrics = evaluate_and_log(
                        "xgboost", y_true, xgb_out["y_pred"], xgb_out["y_proba"],
                        xgb_out["classes"], df_test, target_col
                    )
                    row.update(metrics)

                    pred_path = (
                        PROJECT_ROOT / f"outputs/predictions/xgboost_ratio{ratio:.2f}_seed{seed}.parquet"
                    )
                    df_pred = df_test.copy()
                    df_pred["predicted_mode"] = xgb_out["y_pred"]
                    df_pred.to_parquet(pred_path, index=False)
                    logger.info(f"XGBoost acc={metrics.get('xgboost_accuracy', 0):.3f} f1={metrics.get('xgboost_macro_f1', 0):.3f}")
                except Exception as e:
                    logger.error(f"XGBoost failed: {e}")

            all_results.append(row)

    # Save summary
    results_df = pd.DataFrame(all_results)
    out_path = PROJECT_ROOT / "outputs/metrics/baseline_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")

    # Print summary table
    key_cols = [
        "fewshot_ratio", "seed",
        "mnl_accuracy", "mnl_macro_f1", "mnl_mode_share_mae",
        "xgboost_accuracy", "xgboost_macro_f1", "xgboost_mode_share_mae",
    ]
    display_cols = [c for c in key_cols if c in results_df.columns]
    if display_cols:
        logger.info("\n--- Results Summary ---")
        logger.info(results_df[display_cols].to_string(index=False, float_format="{:.3f}".format))


def parse_args():
    parser = argparse.ArgumentParser(description="Run TRAIL baseline models")
    parser.add_argument("--model", type=str, default="all",
                        choices=["mnl", "xgboost", "all"],
                        help="Which baseline to run")
    parser.add_argument("--task", type=str, default="mode_choice",
                        help="Task name (mode_choice)")
    parser.add_argument("--fewshot_ratio", type=float, default=None,
                        help="Few-shot ratio (default: run all 0.01/0.05/0.10)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
