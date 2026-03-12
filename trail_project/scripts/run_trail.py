"""
Run the TRAIL policy core for mode choice prediction.

Loads prototypes and historical memory, builds few-shot memory,
runs TRAIL on the 2022 test set, and evaluates results.

Usage:
    python scripts/run_trail.py --fewshot_ratio 0.01 --seed 42
    python scripts/run_trail.py --fewshot_ratio 0.05 --dry_run --n_samples 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import pandas as pd

from trail.data.splitter import split_2022
from trail.evaluation.micro import compute_micro_metrics
from trail.evaluation.macro import compute_macro_metrics
from trail.llm.client import LLMClient
from trail.llm.policy_core import TrailPolicyCore
from trail.prototypes.builder import BehaviorPrototypeMemory
from trail.retrieval.memory_bank import FewShotUpdateMemory
from trail.retrieval.retriever import TrailRetriever
from trail.utils.config import PROJECT_ROOT
from trail.utils.io import load_df, load_pickle
from trail.utils.logging import get_logger
from trail.utils.seed import set_seed

logger = get_logger(
    "run_trail",
    log_file=PROJECT_ROOT / "outputs/logs/run_trail.log",
)


def main(args):
    set_seed(args.seed)
    logger.info(f"=== TRAIL Run: fewshot={args.fewshot_ratio:.2f} seed={args.seed} ===")

    # ---- Load preprocessed data ----
    df_2022 = load_df(PROJECT_ROOT / "data/interim/harmonized/tcs2022_harmonized.parquet")
    logger.info(f"Loaded 2022 data: {len(df_2022):,} records")

    # ---- Split ----
    split = split_2022(df_2022, test_ratio=0.2, fewshot_ratio=args.fewshot_ratio, seed=args.seed)
    df_fewshot = split["few_shot"]
    df_test = split["test"]

    if args.dry_run and args.n_samples:
        df_test = df_test.sample(n=min(args.n_samples, len(df_test)), random_state=args.seed)
        logger.info(f"Dry run: using {len(df_test)} test samples")

    logger.info(f"Few-shot: {len(df_fewshot):,} | Test: {len(df_test):,}")

    # ---- Load prototypes ----
    proto_path = PROJECT_ROOT / "data/processed/prototype_cache/prototypes.pkl"
    if not proto_path.exists():
        raise FileNotFoundError(
            f"Run build_prototypes.py first. Missing: {proto_path}"
        )
    prototype_memory = BehaviorPrototypeMemory.load(proto_path)
    logger.info("Prototypes loaded")

    # ---- Load historical memory ----
    hist_path = PROJECT_ROOT / "data/processed/retrieval_corpus/historical_memory.pkl"
    if not hist_path.exists():
        raise FileNotFoundError(
            f"Run build_prototypes.py first. Missing: {hist_path}"
        )
    hist_memory = load_pickle(hist_path)
    logger.info(f"Historical memory: {hist_memory.size:,} records")

    # ---- Build few-shot memory ----
    fewshot_memory = FewShotUpdateMemory(feature_cols=prototype_memory.feature_cols)
    fewshot_memory.build(df_fewshot, encoder=hist_memory.encoder)
    logger.info(f"Few-shot memory: {fewshot_memory.size:,} records")

    # ---- Build retriever ----
    retriever = TrailRetriever.from_config(
        historical_memory=hist_memory,
        fewshot_memory=fewshot_memory,
        config_name="retriever",
    )

    # ---- Build policy core ----
    client = LLMClient()
    policy = TrailPolicyCore(
        retriever=retriever,
        prototype_memory=prototype_memory,
        client=client,
    )

    # ---- Run predictions ----
    logger.info("Running TRAIL predictions ...")
    pred_df = policy.predict(df_test)

    # Attach to test set
    df_results = df_test.copy()
    df_results["predicted_mode"] = pred_df["predicted_mode"].values
    df_results["confidence"] = pred_df["confidence"].values
    df_results["reasoning_brief"] = pred_df["reasoning_brief"].values
    df_results["prototype_id"] = pred_df["prototype_id"].values

    # Filter valid predictions
    valid_mask = df_results["predicted_mode"] >= 1
    n_valid = valid_mask.sum()
    logger.info(f"Valid predictions: {n_valid}/{len(df_results)}")

    df_eval = df_results[valid_mask].copy()
    y_true = df_eval["main_mode"].values
    y_pred = df_eval["predicted_mode"].values

    # ---- Evaluate ----
    micro = compute_micro_metrics(y_true, y_pred, prefix="trail_")
    macro = compute_macro_metrics(df_eval, predictions_col="predicted_mode", prefix="trail_")
    all_metrics = {**micro, **macro}

    logger.info(
        f"TRAIL results: "
        f"acc={all_metrics.get('trail_accuracy', 0):.3f} "
        f"f1={all_metrics.get('trail_macro_f1', 0):.3f} "
        f"share_MAE={all_metrics.get('trail_mode_share_mae', 0):.4f}"
    )

    # ---- Save results ----
    ratio_str = f"{args.fewshot_ratio:.2f}"
    pred_path = PROJECT_ROOT / f"outputs/predictions/trail_ratio{ratio_str}_seed{args.seed}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(pred_path, index=False)
    logger.info(f"Predictions saved: {pred_path}")

    import json
    metrics_path = PROJECT_ROOT / f"outputs/metrics/trail_ratio{ratio_str}_seed{args.seed}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({**all_metrics, "fewshot_ratio": args.fewshot_ratio, "seed": args.seed}, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    client.log_usage()
    logger.info("=== Done ===")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TRAIL mode choice prediction")
    parser.add_argument("--fewshot_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Run on a small subset for testing")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of test samples in dry run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
