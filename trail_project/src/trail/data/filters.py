"""
Sample filters for the unified harmonized dataset.

Architecture doc prescription:
  - 2011: use full TP24 mechanised-trip table
  - 2022: filter to Main_mode <= 9 (shared 9-class mechanised hierarchy)
  - Both: drop records with missing main_mode or trip_purpose
"""

import pandas as pd

from trail.utils.logging import get_logger

logger = get_logger(__name__)


def filter_valid_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records that are unusable for modeling:
      - missing main_mode (main_mode < 1)
      - missing trip_purpose (trip_purpose < 1)
      - missing age_group (-1)
    """
    before = len(df)
    mask = (
        (df["main_mode"] >= 1) &
        (df["trip_purpose"] >= 1) &
        (df["age_group"] >= 1)
    )
    df_clean = df[mask].copy()
    after = len(df_clean)
    logger.info(f"Valid record filter: {before:,} -> {after:,} (dropped {before - after:,})")
    return df_clean


def filter_2022_main_modes(df_tp_raw: pd.DataFrame,
                            mode_col: str = "Main_mode",
                            max_mode: int = 9) -> pd.DataFrame:
    """
    Apply the 2022 pre-harmonization filter: keep only Main_mode <= max_mode.
    This retains the official 9 major mechanised mode hierarchies and removes
    walk / other / unknown codes (10, 99, 999) before harmonization.
    """
    before = len(df_tp_raw)
    df_filtered = df_tp_raw[df_tp_raw[mode_col] <= max_mode].copy()
    after = len(df_filtered)
    logger.info(
        f"2022 mode filter (Main_mode <= {max_mode}): {before:,} -> {after:,} "
        f"(removed {before - after:,} unknown modes)"
    )
    return df_filtered


def filter_journey_time(df: pd.DataFrame, max_minutes: float = 240.0) -> pd.DataFrame:
    """Remove trips with unrealistically long journey times (>4 hours)."""
    before = len(df)
    df_clean = df[
        df["journey_time"].isna() | (df["journey_time"] <= max_minutes)
    ].copy()
    after = len(df_clean)
    logger.info(f"Journey time filter (<=240 min): {before:,} -> {after:,}")
    return df_clean


def apply_all_filters(df: pd.DataFrame, survey_year: int) -> pd.DataFrame:
    """Apply the full standard filter pipeline for a harmonized dataset."""
    df = filter_valid_records(df)
    df = filter_journey_time(df)
    logger.info(f"TCS {survey_year} final filtered: {len(df):,} records")
    return df
