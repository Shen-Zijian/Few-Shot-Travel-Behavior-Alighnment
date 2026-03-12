"""Raw data loaders for TCS 2011 and TCS 2022."""

from pathlib import Path
from typing import Optional

import pandas as pd

from trail.utils.config import load_data_config, PROJECT_ROOT
from trail.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_data_path(relative_path: str) -> Path:
    """Resolve a data path relative to the project root."""
    return (PROJECT_ROOT / relative_path).resolve()


def _load_access_table(db_path: Path, table_name: str) -> pd.DataFrame:
    """Load an Access table into a DataFrame using pyodbc."""
    try:
        import pyodbc
    except ImportError as exc:
        raise ImportError(
            "pyodbc is required to read the TCS 2011 Access database. "
            "Install it with `pip install pyodbc`."
        ) from exc

    conn_str = rf"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};"
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM [{table_name}]")
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        return pd.DataFrame.from_records(rows, columns=columns)


def load_tcs2011_raw(
    config: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the TCS 2011 HH, HM, TP24, and 11TPUSB tables from Access.

    Returns:
        (df_hh, df_hm, df_tp24, df_tpusb_lookup)
    """
    if config is None:
        config = load_data_config("tcs2011")

    db_path = _resolve_data_path(config["raw_data"]["access_db"])
    if not db_path.exists():
        raise FileNotFoundError(f"TCS 2011 Access database not found at: {db_path}")

    tables = config["tables"]

    logger.info(f"Loading TCS 2011 Access database from {db_path}")
    df_hh = _load_access_table(db_path, tables["household"])
    df_hm = _load_access_table(db_path, tables["member"])
    df_tp = _load_access_table(db_path, tables["trip"])
    df_lookup = _load_access_table(db_path, tables["tpusb_lookup"])

    logger.info(
        "TCS 2011 loaded: "
        f"HH={len(df_hh):,}, HM={len(df_hm):,}, TP24={len(df_tp):,}, 11TPUSB={len(df_lookup):,}"
    )
    return df_hh, df_hm, df_tp, df_lookup


def load_tcs2011_joined(config: Optional[dict] = None) -> pd.DataFrame:
    """
    Load TCS 2011 and join HH + HM + TP24 + 11TPUSB.

    The output is trip-level and already includes:
      - household attributes
      - member attributes
      - origin/destination mapped to the 26-district system
      - household car availability derived from HH vehicle ownership fields
    """
    if config is None:
        config = load_data_config("tcs2011")

    df_hh, df_hm, df_tp, df_lookup = load_tcs2011_raw(config)

    hh_cfg = config["hh_columns"]
    hm_cfg = config["hm_columns"]
    tp_cfg = config["tp_columns"]
    lk_cfg = config["lookup_columns"]

    hh_id = hh_cfg["household_id"]
    hm_hh_id = hm_cfg["household_id"]
    hm_mem_id = hm_cfg["member_id"]
    tp_hh_id = tp_cfg["household_id"]
    tp_mem_id = tp_cfg["member_id"]

    # Household selection and car availability derivation.
    hh_cols = [
        hh_id,
        hh_cfg["hh_size"],
        hh_cfg["hh_income"],
        hh_cfg["hh_weight"],
        hh_cfg["tpusb"],
    ] + hh_cfg["car_cols"]
    hh_cols = [c for c in hh_cols if c in df_hh.columns]
    df_hh_sel = df_hh[hh_cols].copy()

    car_cols = [c for c in hh_cfg["car_cols"] if c in df_hh_sel.columns]
    if car_cols:
        df_hh_sel["_car_count"] = df_hh_sel[car_cols].fillna(0).sum(axis=1)
        df_hh_sel["car_availability_raw"] = df_hh_sel["_car_count"].apply(
            lambda x: 0 if x <= 0 else (1 if x == 1 else 2)
        )
        df_hh_sel.drop(columns=["_car_count"] + car_cols, inplace=True)
    else:
        df_hh_sel["car_availability_raw"] = 0

    # Member selection.
    hm_cols = [
        hm_hh_id,
        hm_mem_id,
        hm_cfg["sex"],
        hm_cfg["age"],
        hm_cfg["employment"],
        hm_cfg["member_weight"],
        hm_cfg.get("urban_rural", "UR_MR"),
    ]
    hm_cols = [c for c in hm_cols if c in df_hm.columns]
    df_hm_sel = df_hm[hm_cols].copy()

    # Trip selection.
    tp_cols = [
        tp_hh_id,
        tp_mem_id,
        tp_cfg["trip_no"],
        tp_cfg["trip_purpose"],
        tp_cfg["main_mode"],
        tp_cfg["depart_time"],
        tp_cfg["arrive_time"],
        tp_cfg["journey_time"],
        tp_cfg["origin_tpusb"],
        tp_cfg["dest_tpusb"],
        tp_cfg["time_period"],
        tp_cfg["trip_weight"],
    ]
    tp_cols = [c for c in tp_cols if c in df_tp.columns]
    df_tp_sel = df_tp[tp_cols].copy()

    # TPUSB -> 26 district lookup for cross-year comparable geography.
    lookup_cols = [lk_cfg["tpusb"], lk_cfg["district26"]]
    df_lookup_sel = df_lookup[lookup_cols].drop_duplicates().copy()

    origin_lookup = df_lookup_sel.rename(
        columns={
            lk_cfg["tpusb"]: tp_cfg["origin_tpusb"],
            lk_cfg["district26"]: tp_cfg["origin_zone"],
        }
    )
    dest_lookup = df_lookup_sel.rename(
        columns={
            lk_cfg["tpusb"]: tp_cfg["dest_tpusb"],
            lk_cfg["district26"]: tp_cfg["dest_zone"],
        }
    )

    logger.info("Joining TP24 + HM ...")
    df = df_tp_sel.merge(
        df_hm_sel,
        on=[tp_hh_id, tp_mem_id],
        how="left",
        suffixes=("", "_hm"),
    )

    logger.info("Joining + HH ...")
    df = df.merge(df_hh_sel, on=hh_id, how="left", suffixes=("", "_hh"))

    logger.info("Joining origin/destination TPUSB -> DB26 ...")
    df = df.merge(origin_lookup, on=tp_cfg["origin_tpusb"], how="left")
    df = df.merge(dest_lookup, on=tp_cfg["dest_tpusb"], how="left")

    logger.info(f"TCS 2011 joined: {len(df):,} trip records")
    return df


def load_tcs2022_raw(config: Optional[dict] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the TCS 2022 HH, HM, and TP tables.

    Returns:
        (df_hh, df_hm, df_tp) - household, member, and trip DataFrames
    """
    if config is None:
        config = load_data_config("tcs2022")

    hh_path = _resolve_data_path(config["raw_data"]["hh_file"])
    hm_path = _resolve_data_path(config["raw_data"]["hm_file"])
    tp_path = _resolve_data_path(config["raw_data"]["tp_file"])

    for p in [hh_path, hm_path, tp_path]:
        if not p.exists():
            raise FileNotFoundError(f"TCS 2022 file not found: {p}")

    logger.info("Loading TCS 2022 HH ...")
    df_hh = pd.read_excel(hh_path, engine="openpyxl")

    logger.info("Loading TCS 2022 HM ...")
    df_hm = pd.read_excel(hm_path, engine="openpyxl")

    logger.info("Loading TCS 2022 TP ...")
    df_tp = pd.read_excel(tp_path, engine="openpyxl")

    logger.info(
        f"TCS 2022 loaded: HH={len(df_hh):,}, HM={len(df_hm):,}, TP={len(df_tp):,}"
    )
    return df_hh, df_hm, df_tp


def load_tcs2022_joined(config: Optional[dict] = None) -> pd.DataFrame:
    """
    Load TCS 2022 and perform the HH + HM + TP join.

    Join keys:
      HH (Q_no) -- HM (Q_no, Mem) -- TP (Q_no, Mem, Trip_no)

    Returns a trip-level DataFrame with household and person attributes attached.
    """
    if config is None:
        config = load_data_config("tcs2022")

    df_hh, df_hm, df_tp = load_tcs2022_raw(config)

    hh_cfg = config["hh_columns"]
    hm_cfg = config["hm_columns"]
    tp_cfg = config["tp_columns"]

    hh_id = hh_cfg["household_id"]
    hm_hh_id = hm_cfg["household_id"]
    hm_mem_id = hm_cfg["member_id"]
    tp_hh_id = tp_cfg["household_id"]
    tp_mem_id = tp_cfg["member_id"]

    # Select relevant HH columns
    hh_cols = [hh_id, hh_cfg["hh_type"], hh_cfg["hh_weight"],
               hh_cfg.get("hh_size", "A3")] + hh_cfg["car_cols"]
    hh_cols = [c for c in hh_cols if c in df_hh.columns]
    df_hh_sel = df_hh[hh_cols].copy()

    # Compute car_availability from car count columns
    car_cols = [c for c in hh_cfg["car_cols"] if c in df_hh.columns]
    if car_cols:
        df_hh_sel["_car_count"] = df_hh_sel[car_cols].fillna(0).sum(axis=1)
        df_hh_sel["car_availability_raw"] = df_hh_sel["_car_count"].apply(
            lambda x: 0 if x == 0 else (1 if x == 1 else 2)
        )
        df_hh_sel.drop(columns=["_car_count"] + car_cols, inplace=True)
    else:
        df_hh_sel["car_availability_raw"] = 0

    # Select relevant HM columns
    hm_cols = [hm_hh_id, hm_mem_id,
               hm_cfg["sex"], hm_cfg["age"],
               hm_cfg["employment"], hm_cfg["member_weight"],
               hm_cfg.get("urban_rural", "UR_MR")]
    hm_cols = [c for c in hm_cols if c in df_hm.columns]
    df_hm_sel = df_hm[hm_cols].copy()

    # Select relevant TP columns
    tp_core = [tp_hh_id, tp_mem_id, tp_cfg["trip_no"],
               tp_cfg["trip_purpose"], tp_cfg["main_mode"],
               tp_cfg["journey_time"], tp_cfg["depart_time"],
               tp_cfg["time_period"], tp_cfg["trip_weight"],
               tp_cfg.get("origin_zone", "O_26PDD"),
               tp_cfg.get("dest_zone", "D_26PDD")]
    tp_core = [c for c in tp_core if c in df_tp.columns]
    df_tp_sel = df_tp[tp_core].copy()

    # Join: TP <- HM on (Q_no, Mem)
    logger.info("Joining TP + HM ...")
    df = df_tp_sel.merge(
        df_hm_sel,
        on=[tp_hh_id, tp_mem_id],
        how="left",
        suffixes=("", "_hm"),
    )

    # Join: result <- HH on Q_no
    logger.info("Joining + HH ...")
    df = df.merge(df_hh_sel, on=hh_id, how="left", suffixes=("", "_hh"))

    logger.info(f"TCS 2022 joined: {len(df):,} trip records")
    return df
