"""
Preprocess TCS 2011: load TP24 + HM + HH, harmonize, filter, save to interim/.

Usage:
    python scripts/preprocess_2011.py
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trail.data.loader import load_tcs2011_joined
from trail.data.harmonizer import harmonize_2011
from trail.data.filters import apply_all_filters
from trail.utils.config import PROJECT_ROOT
from trail.utils.io import save_df
from trail.utils.logging import get_logger

logger = get_logger("preprocess_2011", log_file=PROJECT_ROOT / "outputs/logs/preprocess_2011.log")


def main():
    logger.info("=== TCS 2011 Preprocessing ===")

    # 1. Load and join raw tables
    df_raw = load_tcs2011_joined()

    # 2. Harmonize to unified schema
    df_harm = harmonize_2011(df_raw)

    # 3. Apply filters
    df_clean = apply_all_filters(df_harm, survey_year=2011)

    # 4. Save
    out_path = PROJECT_ROOT / "data/interim/harmonized/tcs2011_harmonized.parquet"
    save_df(df_clean, out_path, fmt="parquet")
    logger.info(f"Saved {len(df_clean):,} records to {out_path}")

    # Print summary statistics
    logger.info("\n--- Mode Distribution (2011) ---")
    mode_counts = df_clean["main_mode"].value_counts().sort_index()
    from trail.data.schema import MODE_LABELS
    for mode_code, count in mode_counts.items():
        pct = count / len(df_clean) * 100
        logger.info(f"  {MODE_LABELS.get(mode_code, '?')} ({mode_code}): {count:,} ({pct:.1f}%)")

    logger.info("\n--- Purpose Distribution (2011) ---")
    from trail.data.schema import PURPOSE_LABELS
    for pur, count in df_clean["trip_purpose"].value_counts().sort_index().items():
        logger.info(f"  {PURPOSE_LABELS.get(pur, '?')} ({pur}): {count:,}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
