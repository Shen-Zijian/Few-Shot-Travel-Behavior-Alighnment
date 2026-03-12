"""
Build behavior prototypes from TCS 2011 historical data.
Also indexes the historical and few-shot memories for retrieval.

Usage:
    python scripts/build_prototypes.py
    python scripts/build_prototypes.py --n_clusters 15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trail.prototypes.builder import BehaviorPrototypeMemory
from trail.retrieval.memory_bank import HistoricalPriorMemory
from trail.utils.config import PROJECT_ROOT
from trail.utils.io import load_df, save_pickle
from trail.utils.logging import get_logger

logger = get_logger(
    "build_prototypes",
    log_file=PROJECT_ROOT / "outputs/logs/build_prototypes.log",
)


def main(args):
    logger.info("=== Building Behavior Prototypes ===")

    # Load 2011 data
    path_2011 = PROJECT_ROOT / "data/interim/harmonized/tcs2011_harmonized.parquet"
    if not path_2011.exists():
        raise FileNotFoundError(f"Run preprocess_2011.py first. Missing: {path_2011}")

    df_2011 = load_df(path_2011)
    logger.info(f"Loaded 2011 data: {len(df_2011):,} records")

    # Build prototypes
    memory = BehaviorPrototypeMemory(
        n_clusters=args.n_clusters,
        random_state=42,
    )
    memory.build(df_2011)

    # Print prototype summaries
    logger.info("\n--- Prototype Summaries ---")
    for p in memory.prototypes:
        logger.info(
            f"[{p['prototype_id']:2d}] n={p['n_samples']:5,} "
            f"label={p['semantic_label']:<50} "
            f"mode_share={p['mode_share']}"
        )

    # Save
    proto_path = PROJECT_ROOT / "data/processed/prototype_cache/prototypes.pkl"
    memory.save(proto_path)
    logger.info(f"Prototypes saved: {proto_path}")

    # Build and save historical prior memory (indexed for retrieval)
    logger.info("Building HistoricalPriorMemory index ...")
    hist_memory = HistoricalPriorMemory(
        feature_cols=memory.feature_cols,
    )
    hist_memory.build(df_2011, encoder=memory.encoder)

    hist_path = PROJECT_ROOT / "data/processed/retrieval_corpus/historical_memory.pkl"
    save_pickle(hist_memory, hist_path)
    logger.info(f"Historical memory saved: {hist_path}")

    logger.info("=== Done ===")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
