"""
Preprocess TCS 2022: load HH+HM+TP, join, filter, harmonize, save to interim/.

Usage:
    python scripts/preprocess_2022.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trail.data.loader import load_tcs2022_joined, load_tcs2022_raw
from trail.data.harmonizer import harmonize_2022
from trail.data.filters import filter_2022_main_modes, apply_all_filters
from trail.utils.config import PROJECT_ROOT, load_data_config
from trail.utils.io import save_df
from trail.utils.logging import get_logger

logger = get_logger("preprocess_2022", log_file=PROJECT_ROOT / "outputs/logs/preprocess_2022.log")


def main():
    logger.info("=== TCS 2022 Preprocessing ===")

    cfg_2022 = load_data_config("tcs2022")

    # 1. Load raw tables
    df_hh, df_hm, df_tp = load_tcs2022_raw(cfg_2022)

    # 2. Filter TP to Main_mode <= 9 BEFORE join (removes 99/999 codes)
    tp_col = cfg_2022["tp_columns"]["main_mode"]
    df_tp_filtered = filter_2022_main_modes(df_tp, mode_col=tp_col, max_mode=9)

    # 3. Perform the HH + HM + TP join using the filtered TP
    # Temporarily replace the TP in a mock joined DataFrame
    logger.info("Joining HH + HM + filtered TP ...")
    import pandas as pd
    hh_cfg = cfg_2022["hh_columns"]
    hm_cfg = cfg_2022["hm_columns"]
    tp_cfg = cfg_2022["tp_columns"]

    hh_id = hh_cfg["household_id"]
    hm_mem = hm_cfg["member_id"]

    # Select HH columns
    car_cols = [c for c in hh_cfg["car_cols"] if c in df_hh.columns]
    hh_sel_cols = [hh_id, hh_cfg["hh_type"], hh_cfg["hh_weight"],
                   hh_cfg.get("hh_size", "A3")] + car_cols
    hh_sel_cols = [c for c in hh_sel_cols if c in df_hh.columns]
    df_hh_sel = df_hh[hh_sel_cols].copy()

    if car_cols:
        df_hh_sel["_car_count"] = df_hh_sel[car_cols].fillna(0).sum(axis=1)
        df_hh_sel["car_availability_raw"] = df_hh_sel["_car_count"].apply(
            lambda x: 0 if x == 0 else (1 if x == 1 else 2)
        )
        df_hh_sel.drop(columns=["_car_count"] + car_cols, inplace=True)
    else:
        df_hh_sel["car_availability_raw"] = 0

    # Select HM columns
    hm_sel_cols = [hm_cfg["household_id"], hm_mem,
                   hm_cfg["sex"], hm_cfg["age"],
                   hm_cfg["employment"], hm_cfg["member_weight"]]
    hm_sel_cols = [c for c in hm_sel_cols if c in df_hm.columns]
    df_hm_sel = df_hm[hm_sel_cols].copy()

    # Select TP columns
    tp_core = [tp_cfg["household_id"], tp_cfg["member_id"], tp_cfg["trip_no"],
               tp_cfg["trip_purpose"], tp_cfg["main_mode"],
               tp_cfg["journey_time"], tp_cfg["depart_time"],
               tp_cfg["time_period"], tp_cfg["trip_weight"],
               tp_cfg.get("origin_zone", "O_26PDD"),
               tp_cfg.get("dest_zone", "D_26PDD")]
    tp_core = [c for c in tp_core if c in df_tp_filtered.columns]
    df_tp_sel = df_tp_filtered[tp_core].copy()

    df_joined = df_tp_sel.merge(
        df_hm_sel,
        on=[tp_cfg["household_id"], tp_cfg["member_id"]],
        how="left",
    )
    df_joined = df_joined.merge(df_hh_sel, on=hh_id, how="left")
    logger.info(f"Joined: {len(df_joined):,} records")

    # 4. Harmonize to unified schema
    df_harm = harmonize_2022(df_joined, cfg_2022)

    # 5. Apply filters
    df_clean = apply_all_filters(df_harm, survey_year=2022)

    # 6. Save
    out_path = PROJECT_ROOT / "data/interim/harmonized/tcs2022_harmonized.parquet"
    save_df(df_clean, out_path, fmt="parquet")
    logger.info(f"Saved {len(df_clean):,} records to {out_path}")

    # Print summary
    logger.info("\n--- Mode Distribution (2022) ---")
    mode_counts = df_clean["main_mode"].value_counts().sort_index()
    from trail.data.schema import MODE_LABELS
    for mode_code, count in mode_counts.items():
        pct = count / len(df_clean) * 100
        logger.info(f"  {MODE_LABELS.get(mode_code, '?')} ({mode_code}): {count:,} ({pct:.1f}%)")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
