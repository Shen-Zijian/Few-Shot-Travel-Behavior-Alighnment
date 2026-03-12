"""
Build the combined harmonized dataset by loading both preprocessed years
and saving the combined file for use in downstream experiments.

Requires preprocess_2011.py and preprocess_2022.py to have been run first.

Usage:
    python scripts/build_harmonized_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from trail.utils.config import PROJECT_ROOT
from trail.utils.io import save_df, load_df
from trail.utils.logging import get_logger

logger = get_logger("build_harmonized", log_file=PROJECT_ROOT / "outputs/logs/build_harmonized.log")


def main():
    logger.info("=== Building Combined Harmonized Dataset ===")

    path_2011 = PROJECT_ROOT / "data/interim/harmonized/tcs2011_harmonized.parquet"
    path_2022 = PROJECT_ROOT / "data/interim/harmonized/tcs2022_harmonized.parquet"

    for p in [path_2011, path_2022]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing preprocessed file: {p}\n"
                f"Run preprocess_2011.py and preprocess_2022.py first."
            )

    df_2011 = load_df(path_2011)
    df_2022 = load_df(path_2022)

    logger.info(f"2011: {len(df_2011):,} records | 2022: {len(df_2022):,} records")

    df_all = pd.concat([df_2011, df_2022], ignore_index=True)

    out_path = PROJECT_ROOT / "data/processed/harmonized_all.parquet"
    save_df(df_all, out_path)
    logger.info(f"Combined dataset saved: {len(df_all):,} records -> {out_path}")

    # Print cross-year mode share comparison
    from trail.data.schema import MODE_LABELS
    logger.info("\n--- Cross-Year Mode Share Comparison ---")
    logger.info(f"{'Mode':<20} {'2011 %':>8} {'2022 %':>8}")
    logger.info("-" * 40)
    all_modes = sorted(set(df_2011["main_mode"].unique()) | set(df_2022["main_mode"].unique()))
    for mode in all_modes:
        pct_2011 = (df_2011["main_mode"] == mode).mean() * 100
        pct_2022 = (df_2022["main_mode"] == mode).mean() * 100
        label = MODE_LABELS.get(mode, f"Mode {mode}")
        logger.info(f"{label:<20} {pct_2011:>7.1f}% {pct_2022:>7.1f}%")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
