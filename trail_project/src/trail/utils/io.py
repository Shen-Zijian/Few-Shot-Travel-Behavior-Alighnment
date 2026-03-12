"""I/O utilities: save/load DataFrames, JSON, pickle."""

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_df(df: pd.DataFrame, path: str | Path, fmt: str = "parquet") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        raise ValueError(f"Unknown format: {fmt}")


def load_df(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    raise ValueError(f"Unknown file type: {path.suffix}")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
