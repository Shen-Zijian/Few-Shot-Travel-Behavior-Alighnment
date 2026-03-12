"""
Categorical feature encoding utilities.

Provides consistent one-hot and ordinal encoders that can be
fit on 2022 training data and applied to test/LLM retrieval.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from trail.data.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


class CategoricalEncoder:
    """
    Fits and transforms categorical columns via ordinal encoding.
    Unknown categories (e.g. from 2011) map to a fixed 'unknown' code.
    """

    def __init__(self, columns: list[str] = None):
        self.columns = columns or CATEGORICAL_COLUMNS
        self.encoders: dict[str, LabelEncoder] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        for col in self.columns:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            values = df[col].fillna(-1).astype(int).astype(str).values
            le.fit(values)
            self.encoders[col] = le
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        result = df.copy()
        for col in self.columns:
            if col not in df.columns or col not in self.encoders:
                continue
            le = self.encoders[col]
            values = df[col].fillna(-1).astype(int).astype(str).values
            # Map unseen categories to 0 (safe fallback)
            known = set(le.classes_)
            encoded = np.array([
                le.transform([v])[0] if v in known else 0
                for v in values
            ])
            result[col] = encoded
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def get_mode_label_counts(df: pd.DataFrame, mode_col: str = "main_mode") -> dict[int, int]:
    """Return counts per mode label (for 2022 training set class balance)."""
    return df[mode_col].value_counts().sort_index().to_dict()


def compute_category_ranges(df: pd.DataFrame, columns: list[str] = None) -> dict[str, list]:
    """Return sorted unique values per categorical column."""
    columns = columns or CATEGORICAL_COLUMNS
    return {col: sorted(df[col].dropna().unique().tolist()) for col in columns if col in df.columns}
