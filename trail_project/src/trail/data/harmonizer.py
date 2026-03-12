"""
Variable harmonizer: maps TCS 2011 and 2022 raw fields to the unified schema.

All mapping rules are driven by configs/data/harmonization.yaml.
"""

from typing import Optional

import numpy as np
import pandas as pd

from trail.utils.config import load_data_config
from trail.utils.logging import get_logger
from trail.data.schema import UNIFIED_COLUMNS

logger = get_logger(__name__)


def _load_harmonization() -> dict:
    return load_data_config("harmonization")


def _map_series(series: pd.Series, mapping: dict, default: int = -1) -> pd.Series:
    """Map a Series using a dict, filling unknowns with default."""
    return series.map(mapping).fillna(default).astype(int)


def _as_id_str(series: pd.Series) -> pd.Series:
    """Convert identifier-like numeric columns to clean integer strings."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(-1).astype(int).astype(str)


def _derive_age_group(age_series: pd.Series) -> pd.Series:
    """Bin raw age integers into age group codes (1-5)."""
    bins = [0, 15, 25, 45, 65, 999]
    labels = [1, 2, 3, 4, 5]
    result = pd.cut(age_series, bins=bins, labels=labels, right=False)
    return result.cat.codes.add(1).where(result.notna(), -1).astype(int)


def _hhmm_to_decimal(hhmm_series: pd.Series) -> pd.Series:
    """Convert HHMM integer (e.g. 830 = 08:30, 1530 = 15:30) to decimal hours."""
    hour = (hhmm_series // 100).fillna(-1)
    minute = (hhmm_series % 100).fillna(0)
    return hour + minute / 60.0


def _compute_journey_time_minutes(depart_hhmm: pd.Series, arrive_hhmm: pd.Series) -> pd.Series:
    """
    Compute journey time in minutes from HHMM departure and arrival times.
    Handles simple cases; returns NaN if either is missing.
    """
    depart_decimal = _hhmm_to_decimal(depart_hhmm)
    arrive_decimal = _hhmm_to_decimal(arrive_hhmm)
    duration = (arrive_decimal - depart_decimal) * 60
    duration = duration.where(duration >= 0, duration + 24 * 60)
    valid = depart_hhmm.notna() & arrive_hhmm.notna()
    return duration.where(valid, other=np.nan)


def _decimal_hour_to_period(depart_series: pd.Series) -> pd.Series:
    """
    Convert 2022 departure time (decimal hours 0-24, e.g. 8.5 = 08:30)
    to unified time period.
    Uses the 4_Pks column when available; this is a fallback from D7.
    """
    conditions = [
        (depart_series >= 7) & (depart_series < 9.5),
        (depart_series >= 9.5) & (depart_series < 16.5),
        (depart_series >= 16.5) & (depart_series < 19),
    ]
    choices = [1, 2, 3]
    period = np.select(conditions, choices, default=4)
    period = np.where(depart_series.isna(), -1, period)
    return pd.Series(period, index=depart_series.index, dtype=int)


def harmonize_2011(
    df_joined: pd.DataFrame,
    cfg_2011: Optional[dict] = None,
    harm_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Map the joined TCS 2011 TP24 + HM + HH + 11TPUSB DataFrame to the unified schema.

    Returns a DataFrame with columns matching UNIFIED_COLUMNS.
    """
    if cfg_2011 is None:
        cfg_2011 = load_data_config("tcs2011")
    if harm_cfg is None:
        harm_cfg = _load_harmonization()

    hh = cfg_2011["hh_columns"]
    hm = cfg_2011["hm_columns"]
    tp = cfg_2011["tp_columns"]
    logger.info("Harmonizing TCS 2011 ...")

    df = pd.DataFrame()

    # --- Identifiers ---
    hh_id = _as_id_str(df_joined[tp["household_id"]])
    mem_id = _as_id_str(df_joined[tp["member_id"]])
    trip_num = pd.to_numeric(df_joined[tp["trip_no"]], errors="coerce").fillna(0).astype(int)
    df["household_id"] = hh_id
    df["person_id"] = hh_id + "_" + mem_id
    df["trip_no"] = trip_num
    df["survey_year"] = 2011
    df["case_id"] = "2011_" + hh_id + "_" + mem_id + "_" + trip_num.astype(str)

    # --- Household attributes ---
    df["hh_size"] = pd.to_numeric(df_joined[hh["hh_size"]], errors="coerce").fillna(-1).astype(int)
    if "car_availability_raw" in df_joined.columns:
        df["car_availability"] = df_joined["car_availability_raw"].fillna(0).astype(int)
    else:
        df["car_availability"] = 0

    # --- Person attributes ---
    age_raw = pd.to_numeric(df_joined[hm["age"]], errors="coerce")
    age_raw = age_raw.where(age_raw < 120, np.nan)
    df["age_group"] = _derive_age_group(age_raw)

    df["sex"] = _map_series(df_joined[hm["sex"]], {1: 1, 2: 2}, default=-1)

    emp_map = {
        int(k): int(v)
        for k, v in harm_cfg["employment_2011_to_unified"].items()
    }
    df["employment_status"] = _map_series(df_joined[hm["employment"]], emp_map, default=7)

    inc_map = {
        int(k): int(v)
        for k, v in harm_cfg["income_2011_to_unified"].items()
    }
    df["income_group"] = _map_series(df_joined[hh["hh_income"]], inc_map, default=0)

    # --- Trip purpose ---
    purpose_map = {
        int(k): int(v)
        for k, v in harm_cfg["purpose_2011_to_unified"].items()
    }
    raw_purpose = pd.to_numeric(df_joined[tp["trip_purpose"]], errors="coerce")
    df["trip_purpose"] = _map_series(raw_purpose, purpose_map, default=7)
    df["raw_purpose_2011"] = raw_purpose.fillna(-1).astype(int).astype(str)

    # --- Main mode ---
    mode_map = {
        int(k): int(v)
        for k, v in harm_cfg["mode_2011_to_unified"].items()
    }
    raw_mode = pd.to_numeric(df_joined[tp["main_mode"]], errors="coerce")
    df["main_mode"] = _map_series(raw_mode, mode_map, default=-1)
    df["raw_mode_2011"] = raw_mode.fillna(-1).astype(int).astype(str)

    # --- Trip context ---
    period_col = tp.get("time_period", "TiPer")
    if period_col in df_joined.columns:
        period_map = {
            int(k): int(v)
            for k, v in harm_cfg["time_period_2011_to_unified"].items()
        }
        df["departure_period"] = _map_series(df_joined[period_col], period_map, default=-1)
    else:
        depart_raw = pd.to_numeric(df_joined[tp["depart_time"]], errors="coerce")
        df["departure_period"] = _decimal_hour_to_period(_hhmm_to_decimal(depart_raw))

    journey_raw = pd.to_numeric(df_joined[tp["journey_time"]], errors="coerce")
    if journey_raw.notna().any():
        df["journey_time"] = journey_raw
    else:
        depart_raw = pd.to_numeric(df_joined[tp["depart_time"]], errors="coerce")
        arrive_raw = pd.to_numeric(df_joined[tp["arrive_time"]], errors="coerce")
        df["journey_time"] = _compute_journey_time_minutes(depart_raw, arrive_raw)

    df["origin_zone"] = pd.to_numeric(
        df_joined.get(tp["origin_zone"]), errors="coerce"
    ).fillna(0).astype(int).astype(str)
    df["destination_zone"] = pd.to_numeric(
        df_joined.get(tp["dest_zone"]), errors="coerce"
    ).fillna(0).astype(int).astype(str)

    # --- Weight ---
    df["trip_weight"] = pd.to_numeric(df_joined[tp["trip_weight"]], errors="coerce").fillna(1.0)

    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    logger.info(f"TCS 2011 harmonized: {len(df):,} records")
    return df[UNIFIED_COLUMNS]


def harmonize_2022(
    df_joined: pd.DataFrame,
    cfg_2022: Optional[dict] = None,
    harm_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Map the joined TCS 2022 DataFrame to the unified schema.

    Input is the output of loader.load_tcs2022_joined().
    Returns a DataFrame with columns matching UNIFIED_COLUMNS.
    """
    if cfg_2022 is None:
        cfg_2022 = load_data_config("tcs2022")
    if harm_cfg is None:
        harm_cfg = _load_harmonization()

    tp = cfg_2022["tp_columns"]
    hm = cfg_2022["hm_columns"]
    hh = cfg_2022["hh_columns"]
    logger.info("Harmonizing TCS 2022 ...")

    df = pd.DataFrame()

    # --- Identifiers ---
    hh_id = _as_id_str(df_joined[tp["household_id"]])
    mem_id = _as_id_str(df_joined[tp["member_id"]])
    trip_num = pd.to_numeric(df_joined[tp["trip_no"]], errors="coerce").fillna(0).astype(int)
    df["household_id"] = hh_id
    df["person_id"] = hh_id + "_" + mem_id
    df["trip_no"] = trip_num
    df["survey_year"] = 2022
    df["case_id"] = "2022_" + hh_id + "_" + mem_id + "_" + trip_num.astype(str)

    # --- Household attributes ---
    hh_size_col = hh.get("hh_size", "A3")
    if hh_size_col in df_joined.columns:
        df["hh_size"] = pd.to_numeric(df_joined[hh_size_col], errors="coerce").fillna(-1).astype(int)
    else:
        df["hh_size"] = -1

    if "car_availability_raw" in df_joined.columns:
        df["car_availability"] = df_joined["car_availability_raw"].fillna(0).astype(int)
    else:
        df["car_availability"] = 0

    # --- Person attributes ---
    age_raw = pd.to_numeric(df_joined[hm["age"]], errors="coerce")
    age_raw = age_raw.where(age_raw < 120, np.nan)
    df["age_group"] = _derive_age_group(age_raw)

    df["sex"] = _map_series(df_joined[hm["sex"]], {1: 1, 2: 2}, default=-1)

    emp_map = {
        int(k): int(v)
        for k, v in harm_cfg["employment_2022_to_unified"].items()
    }
    df["employment_status"] = _map_series(df_joined[hm["employment"]], emp_map, default=7)

    # Income not available for 2022 in current data
    df["income_group"] = 0

    # --- Trip purpose ---
    purpose_map = {
        int(k): int(v)
        for k, v in harm_cfg["purpose_2022_to_unified"].items()
    }
    df["trip_purpose"] = _map_series(df_joined[tp["trip_purpose"]], purpose_map, default=7)
    df["raw_purpose_2011"] = None

    # --- Main mode ---
    mode_map = {
        int(k): int(v)
        for k, v in harm_cfg["mode_2022_to_unified"].items()
    }
    raw_mode = pd.to_numeric(df_joined[tp["main_mode"]], errors="coerce")
    df["main_mode"] = _map_series(raw_mode, mode_map, default=-1)
    df["raw_mode_2011"] = None

    # --- Trip context ---
    period_col = tp.get("time_period", "4_Pks")
    if period_col in df_joined.columns:
        period_map = {
            int(k): int(v)
            for k, v in harm_cfg["time_period_2022_to_unified"].items()
        }
        df["departure_period"] = _map_series(df_joined[period_col], period_map, default=-1)
    else:
        depart_raw = pd.to_numeric(df_joined[tp["depart_time"]], errors="coerce")
        df["departure_period"] = _decimal_hour_to_period(depart_raw)

    df["journey_time"] = pd.to_numeric(df_joined[tp["journey_time"]], errors="coerce")

    origin_col = tp.get("origin_zone", "O_26PDD")
    dest_col = tp.get("dest_zone", "D_26PDD")
    df["origin_zone"] = pd.to_numeric(
        df_joined[origin_col], errors="coerce"
    ).fillna(0).astype(int).astype(str) if origin_col in df_joined.columns else ""
    df["destination_zone"] = pd.to_numeric(
        df_joined[dest_col], errors="coerce"
    ).fillna(0).astype(int).astype(str) if dest_col in df_joined.columns else ""

    # --- Weight ---
    df["trip_weight"] = pd.to_numeric(df_joined[tp["trip_weight"]], errors="coerce").fillna(1.0)

    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    logger.info(f"TCS 2022 harmonized: {len(df):,} records")
    return df[UNIFIED_COLUMNS]
